import os
import tempfile
import time
from datetime import datetime
from typing import Iterable
from urllib.parse import quote_plus, urlparse, parse_qs

import kagglehub
import pandas as pd
import requests
from bs4 import BeautifulSoup

from functions.config.settings import get_settings
from functions.utils.utils import to_date_range
from functions.sql.sql_client import SqlClient


settings = get_settings()

# HTTP headers – pretend to be a normal browser to avoid Cloudflare issues.
RUGBY365_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

### general helpers ###
def get_page_htmls(
    scraping_dates: list[datetime],
    integration_provider: str,
    integration_type: str,
) -> list[str]:
    """
    This function is a dispatcher for the appropriate function to get the page htmls from the given integration provider and type 
    for the given dates. 
    NOTE: If more integration providers for getting page htmls are added, this function will need to be updated to dispatch to the appropriate function.
    Accepts:
    - A list of scraping dates
    - The integration provider
    - The integration type
    Returns:
    - A list of page htmls
    """
    if integration_provider.lower() == "rugby365":
        page_htmls = get_page_htmls_from_rugby365(scraping_dates, integration_type)
    else:
        raise ValueError(f"Unsupported integration provider: {integration_provider!r}")
    return page_htmls


def scrape_values_from_page_htmls(
    page_htmls: list[str],
    scraping_dates: list[datetime],
    integration_provider: str,
    integration_type: str,
) -> pd.DataFrame:
    """
    This function is a dispatcher for the appropriate function to scrape values from the given page htmls.
    NOTE: If more integration providers for scraping values are added, this function will need to be updated to dispatch to the appropriate function.
    Accepts:
    - A list of page htmls (list of strings)
    - A list of scraping dates (list of datetime objects)
    - The integration provider (string)
    - The integration type (string)
    Returns:
    - A pandas DataFrame containing the scraped values (DataFrame)
    """
    if integration_provider.lower() == "rugby365":
        scraped_values = scrape_values_from_page_htmls_from_rugby365_for_dates(
            page_htmls=page_htmls,
            scraping_dates=scraping_dates,
            integration_type=integration_type,
        )
    else:
        raise ValueError(f"Unsupported integration provider: {integration_provider!r}")
    return scraped_values


def scrape_values_for_dates(
    scraping_dates: list[datetime],
    integration_provider: str,
    integration_type: str,
) -> pd.DataFrame:
    """
    This is a wrapper function that gets the page htmls for the given dates and then scrapes the values from the page htmls.
    Accepts:
    - A list of scraping dates (list of datetime objects)
    - The integration provider (string)
    - The integration type (string)
    Returns:
    - A pandas DataFrame containing the scraped values (DataFrame)
    """
    # get the page html for the given dates
    page_htmls = get_page_htmls(scraping_dates, integration_provider, integration_type)
    # scrape the values for the given dates
    scraped_values = scrape_values_from_page_htmls(
        page_htmls, scraping_dates, integration_provider, integration_type
    )
    return scraped_values

def save_scraped_values_to_local_file(
    scraped_values: pd.DataFrame,
    integration_provider: str,
    integration_type: str,
) -> str:
    """
    This function persists scraped values to a temporary CSV file and returns its path as per our ingestion pipeline requirements.
    Accepts:
    - A pandas DataFrame containing the scraped values (DataFrame)
    - The integration provider (string)
    - The integration type (string)
    Returns:
    - The path to the saved file (string)

    The ingestion pipeline will upload this file into Blob Storage.
    """
    # create a temporary file path
    prefix = f"{integration_provider}_{integration_type}_"
    # create a temporary file
    file_descriptor, path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(file_descriptor)
    # save the scraped values to the temporary file
    scraped_values.to_csv(path, index=False)
    return path

# provider_helpers ###
# kaggle_helpers ###
def download_historical_results_from_kaggle() -> tuple[str, str]:
    """
    Download historical results from Kaggle.
    
    Returns:
        local_file_path: The path to the downloaded file.
        integration_dataset: The dataset that was downloaded.
    """
    try:
        # Download the dataset
        integration_dataset = settings.kaggle_dataset
        local_directory = kagglehub.dataset_download(integration_dataset)
        local_file_path = os.path.join(local_directory, "results.csv")
        return local_file_path, integration_dataset
    except Exception as e:
        raise ValueError(f"Error downloading historical results from Kaggle: {e!r}")


# rugby365_helpers ###
def get_page_htmls_from_rugby365(
    scraping_dates: Iterable[datetime],
    integration_type: str,
) -> list[str]:
    """
    This function builds the Rugby365 URL for each date using the 'page-change-date=YYYY|MM|DD' query format that Rugby365 uses.
    Accepts:
    - A list of scraping dates (list of datetime objects)
    - The integration type (string)
    Returns:
    - A list of Rugby365 URLs (list of strings)
    """
    page_htmls: list[str] = []
    base_url = "https://rugby365.com/"
    if integration_type == "results":
        base_url += "results/"
    elif integration_type == "fixtures":
        base_url += "fixtures/"
    else:
        raise ValueError(f"Unsupported integration type: {integration_type!r}")

    for day in scraping_dates:
        raw_value = f"{day.year}|{day.month:02d}|{day.day:02d}"
        encoded_value = quote_plus(raw_value)  # encodes '|' to '%7C'
        page_html = f"{base_url}?page-change-date={encoded_value}"
        page_htmls.append(page_html)

    return page_htmls


def parse_rugby365_games_from_html(
    html: str,
    scraping_dates_set: set[datetime],
    integration_type: str,
) -> list[dict]:
    """
    This function provides the core scraping logic for Rugby365 results/fixtures.
    It parses a Rugby365 results/fixtures page and returns a list of game dictionaries.
    Only games whose header date is in `scraping_dates_set` are returned. This
    prevents us from re-scraping older results that Rugby365 continues to show
    on the page beyond the selected date range.

    Accepts:
    - The HTML of a Rugby365 results/fixtures page (string)
    - A set of scraping dates (set of datetime objects)
    - The integration type (string)

    Returns:
    - A list of game dictionaries (list of dictionaries)
    """
    # parse the html into a BeautifulSoup object
    soup = BeautifulSoup(html, "html.parser")
    # find the games section in the BeautifulSoup object
    games_section = soup.find("section", class_="games-list")
    if not games_section:
        return []

    results: list[dict] = []
    # Normalise target dates to plain date objects once (ignore the time component)
    target_dates = {d if hasattr(d, "year") and not hasattr(d, "hour") else d.date() for d in scraping_dates_set}

    for item in games_section.find_all("div", class_="games-list-item", recursive=False):
        # find the date div in the item 
        date_div = item.find("div", class_="date")
        if not date_div:
            continue

        date_text = date_div.get_text(strip=True)
        # Example format: "Sun Dec 7, 2025"
        try:
            header_date = datetime.strptime(date_text, "%a %b %d, %Y").date()
        except ValueError:
            # If the format changes unexpectedly, skip this block rather than fail
            continue

        # We only want games for dates we explicitly asked for.
        if header_date not in target_dates:
            continue

        # For this header date, iterate competitions
        for comp_div in item.find_all("div", class_="comp", recursive=False):
            comp_name_elem = comp_div.find("h2")
            competition_name = comp_name_elem.get_text(strip=True) if comp_name_elem else None
            competition_id = comp_div.get("data-id")

            # find the games container in the competition div
            games_container = comp_div.find("div", class_="games")
            # if there are no games, continue
            if not games_container:
                continue

            for game_div in games_container.find_all("div", class_="game"):
                # Time / state / venue
                time_block = game_div.find("div", class_="time")
                state = None
                venue = None
                # if there is a time block, find the state and venue
                if time_block:
                    # find the state div in the time block
                    state_elem = time_block.find("div", class_="state")
                    # find the venue div in the time block
                    venue_elem = time_block.find("div", class_="venue")
                    state = state_elem.get_text(strip=True) if state_elem else None
                    venue = venue_elem.get_text(strip=True) if venue_elem else None

                # Teams (prefer the logo alt text for clean names)
                home_logo_img = game_div.select_one("div.logo.home img")
                away_logo_img = game_div.select_one("div.logo.away img")
                home_team = (
                    home_logo_img["alt"].strip()
                    if home_logo_img is not None and home_logo_img.get("alt")
                    else None
                )
                away_team = (
                    away_logo_img["alt"].strip()
                    if away_logo_img is not None and away_logo_img.get("alt")
                    else None
                )

                # Kick-off time
                kickoff_elem = game_div.find("div", class_="game-time")
                kickoff_time = kickoff_elem.get_text(strip=True) if kickoff_elem else None

                # Round and live note
                round_elem = game_div.find("div", class_="round")
                round_text = round_elem.get_text(strip=True) if round_elem else None
                live_note_elem = game_div.find("div", class_="live-note")
                live_note = live_note_elem.get_text(strip=True) if live_note_elem else None

                # Scores (may be blank for fixtures)
                def _parse_score(div_class: str) -> int | None:
                    elem = game_div.find("div", class_=div_class)
                    if not elem:
                        return None
                    text = elem.get_text(strip=True)
                    return int(text) if text.isdigit() else None

                home_score = _parse_score("score home")
                away_score = _parse_score("score away")

                # Match link / id
                link_elem = game_div.find("a", class_="link-box")
                match_link = link_elem["href"].strip() if link_elem and link_elem.get("href") else None
                match_id = None
                if match_link:
                    parsed = urlparse(match_link)
                    qs = parse_qs(parsed.query)
                    # Rugby365 uses the 'g' parameter to identify the match
                    # get the 'g' parameter value
                    g_vals = qs.get("g")
                    # if there is a 'g' parameter value, set the match_id
                    if g_vals:
                        match_id = g_vals[0]
                # add the game to the results list
                results.append(
                    {
                        "provider": "rugby365",
                        "integration_type": integration_type,
                        "match_date": header_date.isoformat(),
                        "competition_id": competition_id,
                        "competition_name": competition_name,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "venue": venue,
                        "kickoff_time_local": kickoff_time,
                        "state": state,
                        "round": round_text,
                        "live_note": live_note,
                        "match_link": match_link,
                        "match_id": match_id,
                    }
                )

    return results


def scrape_values_from_page_htmls_from_rugby365_for_dates(
    page_htmls: list[str],
    scraping_dates: list[datetime],
    integration_type: str,
) -> pd.DataFrame:
    """
    This function is a wrapper function that fetches and scrapes Rugby365 pages for the given dates.
    This function bounds the scraping logic to the given dates and integration type, 
    so that we don't scrape older results that Rugby365 continues to show on the page beyond the selected date range.

    Accepts:
    - A list of page HTMLs (list of strings)
    - A list of scraping dates (list of datetime objects)
    - The integration type (string)

    Returns:
    - A pandas DataFrame containing the scraped games
    """
    all_games: list[dict] = []
    dates_set = set(scraping_dates)
    # loop through each HTML (i.e. each date's page)
    for url in page_htmls:
        # get the page response (use browser-like headers to avoid 520/Cloudflare)
        resp = requests.get(url, headers=RUGBY365_REQUEST_HEADERS, timeout=15)
        # raise an error if the response is not successful
        resp.raise_for_status()
        # parse the page games
        page_games = parse_rugby365_games_from_html(
            html=resp.text,
            scraping_dates_set=dates_set,
            integration_type=integration_type,
        )
        all_games.extend(page_games)

        # add a small delay so that we don't overwhelm the Rugby365 servers and get blocked
        time.sleep(1)

    # if there are no games, return an empty DataFrame
    if not all_games:
        return pd.DataFrame()

    # convert the all_games list to a pandas DataFrame
    scraped_df = pd.DataFrame(all_games)

    return scraped_df

## main_functions ###
def download_historical_results(integration_provider: str) -> tuple[str, str]:
    """
    This is the main function that orchestrates the download of historical results from the given integration provider.
    NOTE: If more integration providers for downloading historical results are added, this function will need to be updated to dispatch to the appropriate function.
    Accepts:
    - The integration provider (string)
    Returns:
    - The path to the downloaded file (string)
    - The integration dataset name (string)
    """
    # delegate to the appropriate function based on the integration provider
    if integration_provider == "kaggle":
        local_file_path, integration_dataset = download_historical_results_from_kaggle()
    else:
        raise ValueError(f"Unsupported integration provider: {integration_provider!r}")

    return local_file_path, integration_dataset

def scrape_values(
    integration_provider: str,
    integration_type: str,
    sql_client: SqlClient,
) -> tuple[str, str]:
    """
    Scrape values from the given integration provider and type, then persist to disk.

    Returns:
        local_file_path: Path to the CSV with scraped values.
        integration_dataset: Logical dataset name used for blob naming.
    """
    # Determine which dates to scrape.
    today = datetime.utcnow()
    last_ingestion_event_created_at = sql_client.get_last_ingestion_event_created_at(
        integration_provider=integration_provider,
        integration_type=integration_type,
    )
    if not last_ingestion_event_created_at:
        # Use a conservative bootstrap start date when nothing has been ingested yet.
        last_ingestion_event_created_at = datetime(2025, 12, 12)

    # get the scraping dates (inclusive range from last_ingestion_event_created_at to today)
    scraping_dates = to_date_range(last_ingestion_event_created_at.date(), today.date())
    # Scrape the values for the selected dates
    scraped_values = scrape_values_for_dates(scraping_dates, integration_provider, integration_type)
    # If there are no scraped values, raise a clear, formatted error message.
    if scraped_values.empty:
        raise ValueError(
            f"No scraped values found for provider={integration_provider} and type={integration_type}, over date range {last_ingestion_event_created_at.date()} to {today.date()}"
        )
    # save the scraped values to a local file
    local_file_path = save_scraped_values_to_local_file(
        scraped_values,
        integration_provider,
        integration_type,
    )

    # Create a dataset name based on the scraped date range
    sorted_dates = sorted(scraping_dates)
    date_range_name = f"{sorted_dates[0].strftime('%Y-%m-%d')}_to_{sorted_dates[-1].strftime('%Y-%m-%d')}"
    integration_dataset = f"{integration_provider}_{integration_type}_{date_range_name}"

    return local_file_path, integration_dataset