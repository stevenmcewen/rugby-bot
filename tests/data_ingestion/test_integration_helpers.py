import os
import types
from datetime import datetime, date

import pandas as pd
import pytest

from functions.data_ingestion import integration_helpers as helpers


## Kaggle helpers #############################################################

def test_download_historical_results_from_kaggle_success(monkeypatch, tmp_path):
    """
    download_historical_results_from_kaggle
    - must call kagglehub with the dataset from settings
    - must return a path ending in results.csv inside the downloaded directory
    """
    fake_dir = tmp_path / "kaggle-dataset"
    fake_dir.mkdir()

    def fake_dataset_download(dataset: str) -> str:
        # Ensure we received the dataset from settings
        assert dataset == "owner/dataset"
        return str(fake_dir)

    monkeypatch.setattr(
        helpers,
        "kagglehub",
        types.SimpleNamespace(dataset_download=fake_dataset_download),
    )
    helpers.settings = types.SimpleNamespace(kaggle_dataset="owner/dataset")

    local_file_path, integration_dataset = helpers.download_historical_results_from_kaggle()

    assert integration_dataset == "owner/dataset"
    assert local_file_path.endswith("results.csv")
    assert str(fake_dir) in local_file_path


def test_download_historical_results_from_kaggle_error(monkeypatch):
    """
    download_historical_results_from_kaggle
    - must wrap underlying errors in a ValueError with a clear message
    """

    def fake_dataset_download(dataset: str) -> str:
        raise RuntimeError("kaggle down")

    monkeypatch.setattr(
        helpers,
        "kagglehub",
        types.SimpleNamespace(dataset_download=fake_dataset_download),
    )
    helpers.settings = types.SimpleNamespace(kaggle_dataset="owner/dataset")

    with pytest.raises(ValueError) as exc:
        helpers.download_historical_results_from_kaggle()

    assert "Error downloading historical results from Kaggle" in str(exc.value)


def test_download_historical_results_valid_provider(monkeypatch):
    """
    download_historical_results
    - must delegate to download_historical_results_from_kaggle for 'kaggle'
    """

    def fake_download_from_kaggle() -> tuple[str, str]:
        return "/tmp/results.csv", "owner/dataset"

    monkeypatch.setattr(
        helpers,
        "download_historical_results_from_kaggle",
        fake_download_from_kaggle,
    )

    path, dataset = helpers.download_historical_results("kaggle")

    assert path == "/tmp/results.csv"
    assert dataset == "owner/dataset"


def test_download_historical_results_invalid_provider():
    """
    download_historical_results
    - must raise ValueError for unsupported providers
    """
    with pytest.raises(ValueError) as exc:
        helpers.download_historical_results("other-provider")

    assert "Unsupported integration provider" in str(exc.value)


## Rugby365 URL helpers #######################################################

def test_get_page_htmls_from_rugby365_results_and_fixtures():
    day1 = datetime(2025, 12, 8)
    day2 = datetime(2025, 12, 9)

    urls_results = helpers.get_page_htmls_from_rugby365([day1, day2], "results")
    urls_fixtures = helpers.get_page_htmls_from_rugby365([day1], "fixtures")

    assert all("results/" in u for u in urls_results)
    assert all("fixtures/" in u for u in urls_fixtures)
    # Ensure dates are encoded as YYYY|MM|DD with '|' URL-encoded
    assert "page-change-date=2025%7C12%7C08" in urls_results[0]
    assert "page-change-date=2025%7C12%7C09" in urls_results[1]


def test_get_page_htmls_from_rugby365_invalid_type():
    with pytest.raises(ValueError):
        helpers.get_page_htmls_from_rugby365([datetime(2025, 12, 8)], "unknown")


def test_get_page_htmls_dispatches_to_rugby365(monkeypatch):
    called = {}

    def fake_get_page_htmls_from_rugby365(scraping_dates, integration_type):
        called["scraping_dates"] = scraping_dates
        called["integration_type"] = integration_type
        return ["url1", "url2"]

    monkeypatch.setattr(helpers, "get_page_htmls_from_rugby365", fake_get_page_htmls_from_rugby365)

    dates = [datetime(2025, 12, 8)]
    result = helpers.get_page_htmls(dates, "rugby365", "results")

    assert result == ["url1", "url2"]
    assert called["scraping_dates"] is dates
    assert called["integration_type"] == "results"


def test_get_page_htmls_invalid_provider():
    with pytest.raises(ValueError):
        helpers.get_page_htmls([datetime(2025, 12, 8)], "other", "results")


## Rugby365 HTML parsing ######################################################

def test_parse_rugby365_games_from_html_parses_single_game():
    target_date = date(2025, 12, 8)
    scraping_dates_set = {datetime.combine(target_date, datetime.min.time())}

    html = """
    <section class="games-list">
      <div class="games-list-item">
        <div class="date">Mon Dec 08, 2025</div>
        <div class="comp" data-id="comp-123">
          <h2>URC</h2>
          <div class="games">
            <div class="game">
              <div class="time">
                <div class="state">FT</div>
                <div class="venue">Cape Town Stadium</div>
              </div>
              <div class="logo home"><img alt="Stormers"/></div>
              <div class="logo away"><img alt="Leinster"/></div>
              <div class="game-time">19:00</div>
              <div class="round">Round 5</div>
              <div class="live-note">Full time</div>
              <div class="score home">20</div>
              <div class="score away">18</div>
              <a class="link-box" href="https://rugby365.com/match?g=1234"></a>
            </div>
          </div>
        </div>
      </div>
    </section>
    """

    games = helpers.parse_rugby365_games_from_html(
        html=html,
        scraping_dates_set=scraping_dates_set,
        integration_type="results",
    )

    assert len(games) == 1
    game = games[0]
    assert game["provider"] == "rugby365"
    assert game["integration_type"] == "results"
    assert game["match_date"] == "2025-12-08"
    assert game["competition_id"] == "comp-123"
    assert game["competition_name"] == "URC"
    assert game["home_team"] == "Stormers"
    assert game["away_team"] == "Leinster"
    assert game["home_score"] == 20
    assert game["away_score"] == 18
    assert game["venue"] == "Cape Town Stadium"
    assert game["kickoff_time_local"] == "19:00"
    assert game["state"] == "FT"
    assert game["round"] == "Round 5"
    assert game["live_note"] == "Full time"
    assert game["match_link"].endswith("?g=1234")
    assert game["match_id"] == "1234"


def test_parse_rugby365_games_from_html_skips_non_matching_dates():
    # Header date not in scraping_dates_set should be ignored
    scraping_dates_set = {datetime(2025, 12, 9)}

    html = """
    <section class="games-list">
      <div class="games-list-item">
        <div class="date">Mon Dec 08, 2025</div>
      </div>
    </section>
    """

    games = helpers.parse_rugby365_games_from_html(
        html=html,
        scraping_dates_set=scraping_dates_set,
        integration_type="fixtures",
    )

    assert games == []


def test_parse_rugby365_games_from_html_no_games_section():
    games = helpers.parse_rugby365_games_from_html(
        html="<html><body><p>No games here</p></body></html>",
        scraping_dates_set={datetime(2025, 12, 8)},
        integration_type="results",
    )

    assert games == []


## Rugby365 scrape orchestration #############################################

def test_scrape_values_from_page_htmls_from_rugby365_for_dates_happy_path(monkeypatch):
    """
    scrape_values_from_page_htmls_from_rugby365_for_dates
    - must call requests.get with Rugby365 headers
    - must call parse_rugby365_games_from_html and return a DataFrame of its results
    """
    called = {"urls": [], "parsed": []}

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        called["urls"].append((url, headers, timeout))
        return FakeResponse("<html>ok</html>")

    def fake_parse(html: str, scraping_dates_set, integration_type: str):
        called["parsed"].append((html, scraping_dates_set, integration_type))
        return [
            {
                "provider": "rugby365",
                "integration_type": integration_type,
                "match_date": "2025-12-08",
                "home_team": "Stormers",
                "away_team": "Leinster",
            }
        ]

    monkeypatch.setattr(helpers, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(helpers, "parse_rugby365_games_from_html", fake_parse)

    dates = [datetime(2025, 12, 8)]
    df = helpers.scrape_values_from_page_htmls_from_rugby365_for_dates(
        page_htmls=["https://rugby365.com/results/?page-change-date=x"],
        scraping_dates=dates,
        integration_type="results",
    )

    assert not df.empty
    assert df.iloc[0]["home_team"] == "Stormers"
    assert called["urls"][0][1] == helpers.RUGBY365_REQUEST_HEADERS
    assert called["urls"][0][2] == 15
    assert called["parsed"][0][2] == "results"


def test_scrape_values_from_page_htmls_from_rugby365_for_dates_no_games(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        class FakeResponse:
            text = "<html>no games</html>"

            def raise_for_status(self):
                pass

        return FakeResponse()

    def fake_parse(html: str, scraping_dates_set, integration_type: str):
        return []

    monkeypatch.setattr(helpers, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(helpers, "parse_rugby365_games_from_html", fake_parse)

    df = helpers.scrape_values_from_page_htmls_from_rugby365_for_dates(
        page_htmls=["https://rugby365.com/results/?page-change-date=x"],
        scraping_dates=[datetime(2025, 12, 8)],
        integration_type="results",
    )

    assert df.empty


def test_scrape_values_from_page_htmls_dispatches_to_rugby365(monkeypatch):
    called = {}

    def fake_rugby365(page_htmls, scraping_dates, integration_type):
        called["page_htmls"] = page_htmls
        called["scraping_dates"] = scraping_dates
        called["integration_type"] = integration_type
        return pd.DataFrame([{"provider": "rugby365"}])

    monkeypatch.setattr(
        helpers,
        "scrape_values_from_page_htmls_from_rugby365_for_dates",
        fake_rugby365,
    )

    dates = [datetime(2025, 12, 8)]
    df = helpers.scrape_values_from_page_htmls(
        page_htmls=["url1", "url2"],
        scraping_dates=dates,
        integration_provider="rugby365",
        integration_type="results",
    )

    assert not df.empty
    assert called["page_htmls"] == ["url1", "url2"]
    assert called["scraping_dates"] is dates
    assert called["integration_type"] == "results"


def test_scrape_values_from_page_htmls_invalid_provider():
    with pytest.raises(ValueError):
        helpers.scrape_values_from_page_htmls(
            page_htmls=["url"],
            scraping_dates=[datetime(2025, 12, 8)],
            integration_provider="other",
            integration_type="results",
        )


def test_scrape_values_for_dates_calls_helpers(monkeypatch):
    captured = {}

    def fake_get_page_htmls(scraping_dates, integration_provider, integration_type):
        captured["scraping_dates"] = scraping_dates
        captured["integration_provider"] = integration_provider
        captured["integration_type"] = integration_type
        return ["url1"]

    def fake_scrape_values_from_page_htmls(
        page_htmls, scraping_dates, integration_provider, integration_type
    ):
        captured["page_htmls"] = page_htmls
        captured["scraping_dates_again"] = scraping_dates
        return pd.DataFrame([{"ok": True}])

    monkeypatch.setattr(helpers, "get_page_htmls", fake_get_page_htmls)
    monkeypatch.setattr(helpers, "scrape_values_from_page_htmls", fake_scrape_values_from_page_htmls)

    dates = [datetime(2025, 12, 8)]
    df = helpers.scrape_values_for_dates(
        scraping_dates=dates,
        integration_provider="rugby365",
        integration_type="results",
    )

    assert not df.empty
    assert captured["scraping_dates"] is dates
    assert captured["integration_provider"] == "rugby365"
    assert captured["integration_type"] == "results"
    assert captured["page_htmls"] == ["url1"]
    assert captured["scraping_dates_again"] is dates


def test_save_scraped_values_to_local_file_round_trips_dataframe(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"provider": "rugby365", "home_team": "Stormers", "away_team": "Leinster"},
        ]
    )

    # Force mkstemp to create the file in a temporary test directory with a
    # real OS file descriptor so that os.close(fd) in the implementation works.
    def fake_mkstemp(prefix: str, suffix: str):
        path = tmp_path / f"{prefix}123{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        return fd, str(path)

    monkeypatch.setattr(helpers.tempfile, "mkstemp", fake_mkstemp)

    path = helpers.save_scraped_values_to_local_file(
        scraped_values=df,
        integration_provider="rugby365",
        integration_type="results",
    )

    reloaded = pd.read_csv(path)
    assert list(reloaded["home_team"]) == ["Stormers"]
    assert list(reloaded["away_team"]) == ["Leinster"]


def test_scrape_values_happy_path(monkeypatch):
    """
    scrape_values
    - must derive date range from last_ingestion_event_created_at when present
    - must call scrape_values_for_dates and persist the resulting dataframe
    """
    last_created_at = datetime(2025, 12, 5, 10, 0, 0)

    class FakeSqlClient:
        def __init__(self):
            self.calls = []

        def get_last_ingestion_event_created_at(self, integration_provider, integration_type):
            self.calls.append((integration_provider, integration_type))
            return last_created_at

    sql_client = FakeSqlClient()

    # Capture start/end passed to to_date_range but still use the real implementation
    real_to_date_range = helpers.to_date_range
    captured = {}

    def fake_to_date_range(start_date, end_date):
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return real_to_date_range(start_date, end_date)

    monkeypatch.setattr(helpers, "to_date_range", fake_to_date_range)

    def fake_scrape_values_for_dates(scraping_dates, integration_provider, integration_type):
        # Non‑empty dataframe so that scrape_values does not raise
        return pd.DataFrame([{"provider": integration_provider, "integration_type": integration_type}])

    monkeypatch.setattr(helpers, "scrape_values_for_dates", fake_scrape_values_for_dates)

    def fake_save_scraped_values_to_local_file(scraped_values, integration_provider, integration_type):
        # Just return a predictable path without touching the filesystem
        return "/tmp/scraped.csv"

    monkeypatch.setattr(helpers, "save_scraped_values_to_local_file", fake_save_scraped_values_to_local_file)

    path, dataset = helpers.scrape_values(
        integration_provider="rugby365",
        integration_type="results",
        sql_client=sql_client,
    )

    assert path == "/tmp/scraped.csv"
    # Ensure we used the last_ingestion_event_created_at date as the start of the range
    assert captured["start_date"] == last_created_at.date()
    # End date should be "today" in UTC terms
    assert isinstance(captured["end_date"], date)
    # Integration dataset name should incorporate the computed date range
    expected_prefix = "rugby365_results_"
    assert dataset.startswith(expected_prefix)


def test_scrape_values_raises_when_no_data(monkeypatch):
    class FakeSqlClient:
        def get_last_ingestion_event_created_at(self, integration_provider, integration_type):
            return datetime(2025, 12, 5)

    def fake_to_date_range(start_date, end_date):
        return [start_date, end_date]

    monkeypatch.setattr(helpers, "to_date_range", fake_to_date_range)

    def fake_scrape_values_for_dates(scraping_dates, integration_provider, integration_type):
        # Explicitly return an empty dataframe
        return pd.DataFrame()

    monkeypatch.setattr(helpers, "scrape_values_for_dates", fake_scrape_values_for_dates)

    sql_client = FakeSqlClient()

    with pytest.raises(ValueError) as exc:
        helpers.scrape_values(
            integration_provider="rugby365",
            integration_type="results",
            sql_client=sql_client,
        )

    assert "No scraped values found" in str(exc.value)

