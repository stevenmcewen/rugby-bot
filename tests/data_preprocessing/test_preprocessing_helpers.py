from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from functions.data_preprocessing import preprocessing_helpers as helpers

# create a dummy preprocessing event for testing
class DummyPreprocessingEvent:
    def __init__(
        self,
        *,
        container_name: str = "raw",
        blob_path: str = "path/file.csv",
        integration_provider: str = "kaggle",
        integration_type: str = "historical_results",
        target_table: str = "dbo.InternationalMatchResults",
        source_table: str | None = None,
        batch_id=None,
        system_event_id=None,
    ):
        self.id = uuid4()
        self.container_name = container_name
        self.blob_path = blob_path
        self.integration_provider = integration_provider
        self.integration_type = integration_type
        self.target_table = target_table
        self.source_table = source_table
        # Only needed for stage-2 SQL reads when the source table supports filtering.
        self.batch_id = batch_id or uuid4()
        self.system_event_id = system_event_id or uuid4()

# Does the validate_source_data function raise a ValueError if the source data is empty?
def test_validate_source_data_empty_raises():
    schema = {"columns": ["a"], "data_types": {"a": "int"}, "required": {"a": True}}
    with pytest.raises(ValueError):
        helpers.validate_source_data(pd.DataFrame(), schema)

# Does the validate_source_data function raise a ValueError if a required column is missing?
def test_validate_source_data_missing_required_column_raises():
    schema = {"columns": ["a"], "data_types": {"a": "int"}, "required": {"a": True}}
    with pytest.raises(ValueError) as exc:
        helpers.validate_source_data(pd.DataFrame({"b": [1]}), schema)
    assert "missing required column" in str(exc.value).lower()

# Does the validate_source_data function raise a ValueError if a column has the wrong data type?
def test_validate_source_data_type_mismatch_raises(monkeypatch):
    schema = {"columns": ["a"], "data_types": {"a": "int"}, "required": {"a": True}}
    df = pd.DataFrame({"a": ["not-int"]})
    monkeypatch.setattr(helpers, "matches_type", lambda *_a, **_k: False)
    with pytest.raises(ValueError) as exc:
        helpers.validate_source_data(df, schema)
    assert "unexpected data type" in str(exc.value).lower()

# Does the validate_transformed_data function raise a ValueError if the transformed data is empty?
def test_validate_transformed_data_empty_raises():
    schema = {"columns": ["a"], "data_types": {"a": "int"}, "required": {"a": True}}
    with pytest.raises(ValueError):
        helpers.validate_transformed_data(pd.DataFrame(), schema)

# Does the get_source_schema function build the expected dictionary?
def test_get_source_schema_builds_expected_dict():
    event = DummyPreprocessingEvent(integration_provider="rugby365", integration_type="results")
    # create a fake sql client that returns the expected schema
    class FakeSqlClient:
        def get_schema(self, **kwargs):
            assert kwargs["integration_provider"] == "rugby365"
            assert kwargs["integration_type"] == "results"
            return [
                {"column_name": "a", "data_type": "int", "is_required": 1},
                {"column_name": "b", "data_type": "string", "is_required": 0},
            ]

    schema = helpers.get_source_schema(event, FakeSqlClient())
    assert schema["columns"] == ["a", "b"]
    assert schema["data_types"]["a"] == "int"
    assert schema["required"]["a"] is True
    assert schema["required"]["b"] is False

# Does the get_target_schema function build the expected dictionary?
def test_get_target_schema_builds_expected_dict():
    event = DummyPreprocessingEvent(target_table="dbo.Target")

    class FakeSqlClient:
        def get_schema(self, **kwargs):
            assert kwargs["table_name"] == "dbo.Target"
            return [
                {"column_name": "c", "data_type": "date", "is_required": True},
            ]

    schema = helpers.get_target_schema(event, FakeSqlClient())
    assert schema["columns"] == ["c"]
    assert schema["data_types"]["c"] == "date"
    assert schema["required"]["c"] is True


def test_get_source_schema_uses_source_table_when_provided():
    event = DummyPreprocessingEvent(source_table="dbo.Source")

    class FakeSqlClient:
        def get_schema(self, **kwargs):
            assert kwargs["table_name"] == "dbo.Source"
            return [
                {"column_name": "x", "data_type": "int", "is_required": 1},
            ]

    schema = helpers.get_source_schema(event, FakeSqlClient())
    assert schema["columns"] == ["x"]
    assert schema["data_types"]["x"] == "int"
    assert schema["required"]["x"] is True

# Does the get_source_data function download the blob and read the csv?
def test_get_source_data_downloads_blob_and_reads_csv(monkeypatch):
    event = DummyPreprocessingEvent(container_name="raw", blob_path="x.csv")

    csv_bytes = b"col1,col2\n1,2\n3,4\n"

    # create a fake download stream that returns the expected csv bytes
    class FakeDownloadStream:
        def readall(self):
            return csv_bytes

    # create a fake blob client that returns the expected download stream
    class FakeBlobClient:
        def download_blob(self):
            return FakeDownloadStream()

    # create a fake from connection string function that returns the expected blob client
    def fake_from_connection_string(conn_str: str, container_name: str, blob_name: str):
        assert container_name == "raw"
        assert blob_name == "x.csv"
        assert conn_str == "UseDevelopmentStorage=true"
        return FakeBlobClient()

    # Patch module-level settings and BlobClient factory
    helpers.settings = SimpleNamespace(storage_connection="UseDevelopmentStorage=true")
    monkeypatch.setattr(
        helpers,
        "BlobClient",
        SimpleNamespace(from_connection_string=fake_from_connection_string),
    )

    # call the get_source_data function and check the expected columns and shape
    df = helpers.get_source_data(event, sql_client=SimpleNamespace())
    assert list(df.columns) == ["col1", "col2"]
    assert df.shape == (2, 2)


def test_get_source_data_reads_from_sql_when_source_table_provided():
    batch_id = uuid4()
    system_event_id = uuid4()
    event = DummyPreprocessingEvent(
        source_table="dbo.Source",
        batch_id=batch_id,
        system_event_id=system_event_id,
    )

    expected_df = pd.DataFrame([{"a": 1}])

    class FakeSqlClient:
        def get_schema(self, **kwargs):
            assert kwargs["table_name"] == "dbo.Source"
            # Provide filters so helper will build WHERE + params
            return [
                {"column_name": "batch_id", "data_type": "uniqueidentifier", "is_required": 0},
                {"column_name": "system_event_id", "data_type": "uniqueidentifier", "is_required": 0},
                {"column_name": "a", "data_type": "int", "is_required": 0},
            ]

        def read_table_to_dataframe(self, *, table_name, where_sql=None, params=None, columns=None):
            assert table_name == "dbo.Source"
            assert columns is None
            assert where_sql == "batch_id = :batch_id AND system_event_id = :system_event_id"
            assert params == {"batch_id": str(batch_id), "system_event_id": str(system_event_id)}
            return expected_df

    out = helpers.get_source_data(event, sql_client=FakeSqlClient())
    assert out.equals(expected_df)

# Does the write_data_to_target_table function delegate to the sql client?
def test_write_data_to_target_table_delegates_to_sql_client():
    event = DummyPreprocessingEvent(target_table="dbo.Target")
    df = pd.DataFrame([{"a": 1}])
    # create a fake sql client that returns the expected columns and shape

    class FakeSqlClient:
        def __init__(self):
            self.called_with = None

        def write_dataframe_to_table(self, **kwargs):
            self.called_with = kwargs

    # call the write_data_to_target_table function and check the expected columns and shape
    sql = FakeSqlClient()
    helpers.write_data_to_target_table(df, event, sql)
    assert sql.called_with["df"].equals(df)
    assert sql.called_with["table_name"] == "dbo.Target"


def test_truncate_target_table_delegates_to_sql_client():
    event = DummyPreprocessingEvent(target_table="dbo.Target")

    class FakeSqlClient:
        def __init__(self):
            self.called_with = None

        def truncate_table(self, *, table_name: str):
            self.called_with = table_name

    sql = FakeSqlClient()
    helpers.truncate_target_table(event, sql)
    assert sql.called_with == "dbo.Target"

# Does the add_kickoff_datetime_from_date_and_time function combine the date and time correctly?
def test_add_kickoff_datetime_from_date_and_time_combines_correctly():
    df = pd.DataFrame(
        [
            {"match_date": "2025-12-08", "kickoff_time_local": "4:05AM"},
            {"match_date": "2025-12-08", "kickoff_time_local": "7:30PM"},
        ]
    )
    out = helpers.add_kickoff_datetime_from_date_and_time(
        df,
        date_column="match_date",
        time_column="kickoff_time_local",
        target_column="dt",
    )
    assert pd.to_datetime(out.loc[0, "dt"]) == pd.Timestamp("2025-12-08 04:05:00")
    assert pd.to_datetime(out.loc[1, "dt"]) == pd.Timestamp("2025-12-08 19:30:00")

# Does the add_city_and_country_to_dataframe function map the venue to the city and country correctly?
def test_add_city_and_country_to_dataframe_maps_case_insensitively():
    df = pd.DataFrame([{"venue": " Cape Town Stadium "}])
    # create a fake sql client that returns the expected venue database
    class FakeSqlClient:
        def get_venue_database(self):
            return pd.DataFrame(
                [
                    {"venue": "CAPE TOWN STADIUM", "city": "Cape Town", "country": "South Africa"},
                ]
            )

    # call the add_city_and_country_to_dataframe function and check the expected city and country
    out = helpers.add_city_and_country_to_dataframe(
        df,
        FakeSqlClient(),
        venue_column="venue",
        city_column="city",
        country_column="country",
    )
    assert out.loc[0, "city"] == "Cape Town"
    assert out.loc[0, "country"] == "South Africa"

# Does the determine_neutral_match_international function mark the neutral match correctly?
def test_determine_neutral_match_international_marks_neutral_and_handles_unknown_country():
    # create a dataframe with the expected home team, away team and country
    df = pd.DataFrame(
        [
            {"home_team": "France", "away_team": "Wales", "country": "Italy"},
            {"home_team": "France", "away_team": "Wales", "country": "France"},
            {"home_team": "France", "away_team": "Wales", "country": None},
        ]
    )
    # call the determine_neutral_match_international function and check the expected neutral column
    out = helpers.determine_neutral_match_international(
        df,
        home_team_column="home_team",
        away_team_column="away_team",
        country_column="country",
        neutral_column="neutral",
    )
    # pandas often returns numpy.bool_ scalars; avoid `is True/False`
    assert bool(out.loc[0, "neutral"]) is True
    assert bool(out.loc[1, "neutral"]) is False
    assert bool(out.loc[2, "neutral"]) is False

# Does the determine_world_cup_match function mark the world cup match correctly?
def test_determine_world_cup_match_case_insensitive_contains():
    # create a dataframe with the expected competition name
    df = pd.DataFrame(
        [{"competition_name": "Rugby World Cup 2027"}, {"competition_name": "Six Nations"}]
    )
    out = helpers.determine_world_cup_match(
        df,
        world_cup_column="world_cup",
        competition_name_column="competition_name",
    )
    assert bool(out.loc[0, "world_cup"]) is True
    assert bool(out.loc[1, "world_cup"]) is False

# Does the transform_kaggle_historical_data_to_international_results function transform the Kaggle historical data to international results correctly?
def test_transform_kaggle_historical_data_to_international_results_transforms_and_filters():
    event = DummyPreprocessingEvent(integration_provider="kaggle", integration_type="historical_results")
    # include a duplicate and a row with missing required field to ensure drop/duplicates behavior
    # create a dataframe with the expected data
    source = pd.DataFrame(
        [
            {
                "date": "2025-12-08",
                "home_team": "Stormers",
                "away_team": "Leinster",
                "home_score": 20,
                "away_score": 18,
                "competition": "URC",
                "stadium": "Cape Town Stadium",
                "city": "Cape Town",
                "country": "South Africa",
                "neutral": 0,
                "world_cup": 0,
            },
            {
                "date": "2025-12-08",
                "home_team": "Stormers",
                "away_team": "Leinster",
                "home_score": 20,
                "away_score": 18,
                "competition": "URC",
                "stadium": "Cape Town Stadium",
                "city": "Cape Town",
                "country": "South Africa",
                "neutral": 0,
                "world_cup": 0,
            },
            {
                "date": None,
                "home_team": "X",
                "away_team": "Y",
                "home_score": 1,
                "away_score": 2,
                "competition": "URC",
                "stadium": "S",
                "city": "C",
                "country": "Z",
                "neutral": 1,
                "world_cup": 0,
            },
        ]
    )

    out = helpers.transform_kaggle_historical_data_to_international_results(
        source, event, sql_client=SimpleNamespace()
    )
    assert not out.empty
    assert list(out.columns) == [
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "HomeScore",
        "AwayScore",
        "CompetitionName",
        "Venue",
        "City",
        "Country",
        "Neutral",
        "WorldCup",
    ]
    # Duplicate should have been dropped and invalid row dropped -> 1 row
    assert len(out) == 1
    assert out.loc[0, "HomeTeam"] == "STORMERS"
    assert out.loc[0, "AwayTeam"] == "LEINSTER"
    assert out.loc[0, "CompetitionName"] == "URC"
    assert out.loc[0, "Neutral"] in (False, True)

# Does the transform_rugby365_results_data_to_international_results function filter and enrich the Rugby365 results data correctly?
def test_transform_rugby365_results_data_to_international_results_filters_and_enriches():
    event = DummyPreprocessingEvent(integration_provider="rugby365", integration_type="results")
    source = pd.DataFrame(
        [
            {
                "match_date": "2025-12-08",
                "competition_name": "Six Nations",
                "home_team": "France",
                "away_team": "Wales",
                "home_score": 10,
                "away_score": 7,
                "venue": "Stade de France",
                "kickoff_time_local": "7:30PM",
            },
            {
                "match_date": "2025-12-08",
                "competition_name": "Club Friendly",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 2,
                "venue": "Somewhere",
                "kickoff_time_local": "7:30PM",
            },
        ]
    )
    # create a fake sql client that returns the expected venue database
    class FakeSqlClient:
        def get_venue_database(self):
            return pd.DataFrame(
                [
                    {"venue": "STADE DE FRANCE", "city": "Paris", "country": "France"},
                ]
            )

    out = helpers.transform_rugby365_results_data_to_international_results(
        source, event, FakeSqlClient()
    )
    assert len(out) == 1
    assert out.loc[0, "CompetitionName"] == "SIX NATIONS"
    assert out.loc[0, "City"] == "PARIS"
    assert out.loc[0, "Country"] == "FRANCE"
    assert bool(out.loc[0, "Neutral"]) is False
    assert bool(out.loc[0, "WorldCup"]) is False

# Does the transform_rugby365_fixtures_data_to_international_fixtures function build the transformed data correctly?
def test_transform_rugby365_fixtures_data_to_international_fixtures_builds_kickoff_datetime():
    event = DummyPreprocessingEvent(integration_provider="rugby365", integration_type="fixtures")
    # create a dataframe with the expected data
    source = pd.DataFrame(
        [
            {
                "match_date": "2025-12-08",
                "competition_name": "Rugby World Cup",
                "home_team": "France",
                "away_team": "Wales",
                "venue": "Stade de France",
                "kickoff_time_local": "4:05AM",
            },
        ]
    )
    # create a fake sql client that returns the expected venue database
    class FakeSqlClient:
        def get_venue_database(self):
            return pd.DataFrame(
                [
                    {"venue": "STADE DE FRANCE", "city": "Paris", "country": "France"},
                ]
            )

    out = helpers.transform_rugby365_fixtures_data_to_international_fixtures(
        source, event, FakeSqlClient()
    )
    # call the transform_rugby365_fixtures_data_to_international_fixtures function and check the expected data
    assert len(out) == 1
    assert pd.to_datetime(out.loc[0, "KickoffTimeLocal"]) == pd.Timestamp("2025-12-08 04:05:00")
    assert bool(out.loc[0, "WorldCup"]) is True


# Does the transform_international_results_to_model_ready_data function build the model ready data correctly?
# must build the model ready data correctly
def test_transform_international_results_to_model_ready_data_builds_model_ready_data():
    """
    Stage-2 transform builds leakage-safe pre-match features + targets.
    This test locks in the contract that it adds tier features and drops scores.
    """
    event = DummyPreprocessingEvent(
        integration_provider="kaggle",
        integration_type="historical_results",
        source_table="dbo.InternationalMatchResults",
        target_table="dbo.InternationalMatchResultsModelData",
    )
    source = pd.DataFrame(
        [
            {
                "ID": 1,
                "MatchDate": "2025-12-08",
                "HomeTeam": "FRANCE",
                "AwayTeam": "WALES",
                "HomeScore": 10,
                "AwayScore": 7,
                "CompetitionName": "SIX NATIONS",
                "Venue": "STADE DE FRANCE",
                "City": "PARIS",
                "Country": "FRANCE",
                "Neutral": False,
                "WorldCup": False,
            }
        ]
    )

    class FakeSqlClient:
        def read_table_to_dataframe(self, *, table_name: str, columns=None, where_sql=None, params=None):
            if table_name == "dbo.InternationalRugbyTeams":
                # Used both for TeamNameStandardization and add_international_features
                return pd.DataFrame(
                    [
                        {"TeamName": "FRANCE", "Tier": 1, "Hemisphere": "NH"},
                        {"TeamName": "WALES", "Tier": 1, "Hemisphere": "NH"},
                    ]
                )
            if table_name == "dbo.RugbyVenues":
                return pd.DataFrame([{"VenueName": "STADE DE FRANCE"}])
            raise AssertionError(f"Unexpected table read: {table_name}")

    out = helpers.transform_international_results_to_model_ready_data(
        source,
        event,
        sql_client=FakeSqlClient(),
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1

    # Scores should be dropped (no leakage)
    assert "HomeScore" not in out.columns
    assert "AwayScore" not in out.columns

    # Tier features should exist
    assert out.loc[0, "Home_Tier"] == 1
    assert out.loc[0, "Away_Tier"] == 1
    assert out.loc[0, "Diff_Tier"] == 0

    # Hemisphere is optional; if present it should be added
    assert out.loc[0, "Home_Hemisphere"] == "NH"
    assert out.loc[0, "Away_Hemisphere"] == "NH"

    # Targets should exist
    assert out.loc[0, "PointDiff"] == 3
    assert out.loc[0, "HomeWin"] == 1

    # Weight should be computed (with single row -> 1.0)
    assert float(out.loc[0, "TimeDecayWeight"]) == 1.0


def test_transform_international_fixtures_to_model_ready_data_keeps_fixtures_and_computes_form_without_scores():
    """
    Stage-2 fixtures transform should:
      - include fixture rows even though scores are missing
      - compute rolling form from prior results only (fixtures shouldn't count as losses)
      - drop score columns to avoid leakage
      - not add targets (HomeWin/PointDiff) for fixtures
    """
    event = DummyPreprocessingEvent(
        integration_provider="rugby365",
        integration_type="fixtures",
        source_table="dbo.InternationalMatchFixtures",
        target_table="dbo.InternationalMatchFixturesModelData",
    )

    # One upcoming fixture for FRANCE (no scores, has KickoffTimeLocal)
    fixtures_source = pd.DataFrame(
        [
            {
                "ID": 100,
                "MatchDate": "2025-12-15",
                "KickoffTimeLocal": "2025-12-15 18:00:00",
                "HomeTeam": "FRANCE",
                "AwayTeam": "WALES",
                "CompetitionName": "INTERNATIONALS",
                "Venue": "STADE DE FRANCE",
                "City": "PARIS",
                "Country": "FRANCE",
                "Neutral": False,
                "WorldCup": False,
            }
        ]
    )

    class FakeSqlClient:
        def read_table_to_dataframe(self, *, table_name: str, columns=None, where_sql=None, params=None):
            if table_name == "dbo.InternationalMatchResults":
                # One prior played match (scores present) so form can be computed.
                return pd.DataFrame(
                    [
                        {
                            "ID": 1,
                            "MatchDate": "2025-12-08",
                            "HomeTeam": "FRANCE",
                            "AwayTeam": "WALES",
                            "HomeScore": 10,
                            "AwayScore": 7,
                            "CompetitionName": "INTERNATIONALS",
                            "Venue": "STADE DE FRANCE",
                            "City": "PARIS",
                            "Country": "FRANCE",
                            "Neutral": False,
                            "WorldCup": False,
                        }
                    ]
                )
            if table_name == "dbo.InternationalRugbyTeams":
                # Used both for TeamNameStandardization and add_international_features
                return pd.DataFrame(
                    [
                        {"TeamName": "FRANCE", "Tier": 1, "Hemisphere": "NH"},
                        {"TeamName": "WALES", "Tier": 1, "Hemisphere": "NH"},
                    ]
                )
            if table_name == "dbo.RugbyVenues":
                return pd.DataFrame([{"VenueName": "STADE DE FRANCE"}])
            raise AssertionError(f"Unexpected table read: {table_name}")

    out = helpers.transform_international_fixtures_to_model_ready_data(
        fixtures_source,
        event,
        sql_client=FakeSqlClient(),
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert out.loc[0, "ID"] == 100
    assert pd.notna(out.loc[0, "KickoffTimeLocal"])

    # Scores should be dropped (no leakage)
    assert "HomeScore" not in out.columns
    assert "AwayScore" not in out.columns

    # Targets should NOT exist for fixtures
    assert "PointDiff" not in out.columns
    assert "HomeWin" not in out.columns

    # Form feature should exist and should reflect the one prior win (not be forced to 0).
    assert "Home_FormWinRate" in out.columns
    assert float(out.loc[0, "Home_FormWinRate"]) == 1.0


# Does the team_name_standardization function filter to canonical teams only?
# must filter to canonical teams only
def test_team_name_standardization_filters_to_canonical_teams_only():
    matches = pd.DataFrame(
        [
            {"ID": 1, "HomeTeam": "France", "AwayTeam": "Wales", "Venue": "X"},
            {"ID": 2, "HomeTeam": "France", "AwayTeam": "Italy", "Venue": "X"},
            {"ID": 3, "HomeTeam": None, "AwayTeam": "Wales", "Venue": "X"},
        ]
    )

    class FakeSqlClient:
        def read_table_to_dataframe(self, *, table_name: str, columns=None, where_sql=None, params=None):
            assert table_name == "dbo.InternationalRugbyTeams"
            assert columns == ["TeamName"]
            return pd.DataFrame([{"TeamName": "FRANCE"}, {"TeamName": "WALES"}])

    out = helpers.team_name_standardization(matches, FakeSqlClient())
    assert out["ID"].tolist() == [1]


# Does the venue_standardisation function filter to canonical venues only?
# must filter to canonical venues only
def test_venue_standardisation_filters_to_canonical_venues_only_case_insensitive():
    matches = pd.DataFrame(
        [
            {"ID": 1, "Venue": " Stade de France "},
            {"ID": 2, "Venue": "Unknown Venue"},
            {"ID": 3, "Venue": None},
        ]
    )

    class FakeSqlClient:
        def read_table_to_dataframe(self, *, table_name: str, columns=None, where_sql=None, params=None):
            assert table_name == "dbo.RugbyVenues"
            assert columns == ["VenueName"]
            return pd.DataFrame([{"VenueName": "STADE DE FRANCE"}])

    out = helpers.venue_standardisation(matches, FakeSqlClient(), table_name="dbo.RugbyVenues", venue_col="Venue")
    assert out["ID"].tolist() == [1]


# Does the add_match_targets function compute the point difference, home win and drop draws by default?
# must compute the point difference, home win and drop draws by default
def test_add_match_targets_computes_point_diff_and_home_win_and_drops_draws_by_default():
    df = pd.DataFrame(
        [
            {"HomeScore": 10, "AwayScore": 7},   # home win
            {"HomeScore": 12, "AwayScore": 12},  # draw
            {"HomeScore": 5, "AwayScore": 9},    # home loss
        ]
    )
    out = helpers.add_match_targets(df, drop_draws=True)
    assert out["PointDiff"].tolist() == [3, -4]
    assert out["HomeWin"].tolist() == [1, 0]


# Does the to_team_match_long_format function create two rows per match and correct perspectives?
# must create two rows per match and correct perspectives
def test_to_team_match_long_format_creates_two_rows_per_match_and_correct_perspectives():
    matches = pd.DataFrame(
        [
            {
                "ID": 1,
                "MatchDate": "2025-12-08",
                "HomeTeam": "FRANCE",
                "AwayTeam": "WALES",
                "HomeScore": 10,
                "AwayScore": 7,
                "CompetitionName": "SIX NATIONS",
            }
        ]
    )
    long_df = helpers.to_team_match_long_format(
        matches,
        id_col="ID",
        date_col="MatchDate",
        home_team_col="HomeTeam",
        away_team_col="AwayTeam",
        home_score_col="HomeScore",
        away_score_col="AwayScore",
        extra_context_cols=("CompetitionName", "MissingCol"),
    )

    assert len(long_df) == 2
    # Home perspective row
    home_row = long_df[long_df["IsHome"] == 1].iloc[0]
    assert home_row["Team"] == "FRANCE"
    assert home_row["Opponent"] == "WALES"
    assert home_row["PointsFor"] == 10
    assert home_row["PointsAgainst"] == 7
    assert home_row["Win"] == 1
    # Away perspective row
    away_row = long_df[long_df["IsHome"] == 0].iloc[0]
    assert away_row["Team"] == "WALES"
    assert away_row["Opponent"] == "FRANCE"
    assert away_row["PointsFor"] == 7
    assert away_row["PointsAgainst"] == 10
    assert away_row["Loss"] == 1
    # Context column preserved when present
    assert "CompetitionName" in long_df.columns


# Does the compute_rolling_team_form function shift so the current match is not included?
# must shift so the current match is not included
def test_compute_rolling_team_form_shifts_so_current_match_not_included():
    team_long = pd.DataFrame(
        [
            # Same team, two matches in order
            {"ID": 1, "MatchDate": "2025-01-01", "Team": "A", "Win": 1, "Draw": 0, "Loss": 0, "PointsFor": 10, "PointsAgainst": 5, "PointDiff": 5},
            {"ID": 2, "MatchDate": "2025-02-01", "Team": "A", "Win": 0, "Draw": 0, "Loss": 1, "PointsFor": 7, "PointsAgainst": 12, "PointDiff": -5},
        ]
    )
    out = helpers.compute_rolling_team_form(team_long, form_window=10, date_col="MatchDate")

    # First match has no prior history
    assert pd.isna(out.loc[0, "FormWinRate"])
    assert pd.isna(out.loc[0, "FormPointsFor"])
    assert pd.isna(out.loc[0, "FormGames"])

    # Second match sees only the first match
    assert out.loc[1, "FormWinRate"] == 1.0
    assert out.loc[1, "FormPointsFor"] == 10.0
    assert int(out.loc[1, "FormGames"]) == 1
    assert str(out["FormGames"].dtype) == "Int64"


# Does the attach_rolling_team_form_to_matches function join home and away and build diffs?
# must join home and away and build diffs
def test_attach_rolling_team_form_to_matches_joins_home_away_and_builds_diffs():
    results_df = pd.DataFrame([{"ID": 1, "HomeTeam": "A", "AwayTeam": "B"}])
    team_form_long = pd.DataFrame(
        [
            {"ID": 1, "Team": "A", "IsHome": 1, "FormWinRate": 0.6, "FormDrawRate": 0.1, "FormPointsFor": 20.0, "FormPointsAgainst": 18.0, "FormPointDiff": 2.0, "FormGames": 5},
            {"ID": 1, "Team": "B", "IsHome": 0, "FormWinRate": 0.4, "FormDrawRate": 0.2, "FormPointsFor": 15.0, "FormPointsAgainst": 19.0, "FormPointDiff": -4.0, "FormGames": 5},
        ]
    )
    out = helpers.attach_rolling_team_form_to_matches(results_df, team_form_long)
    assert out.loc[0, "Home_FormWinRate"] == 0.6
    assert out.loc[0, "Away_FormWinRate"] == 0.4
    # Avoid brittle binary-float equality
    assert float(out.loc[0, "Diff_FormWinRate"]) == pytest.approx(0.2)
    assert out.loc[0, "Diff_FormPointDiff"] == 6.0


# Does the add_international_features function map tier and hemisphere and is non-destructive?
# must map tier and hemisphere and is non-destructive
def test_add_international_features_maps_tier_and_hemisphere_and_is_non_destructive():
    df = pd.DataFrame(
        [
            {"HomeTeam": "FRANCE", "AwayTeam": "WALES"},
            {"HomeTeam": "FRANCE", "AwayTeam": "UNKNOWN"},
        ]
    )

    class FakeSqlClient:
        def read_table_to_dataframe(self, *, table_name: str, columns=None, where_sql=None, params=None):
            assert table_name == "dbo.InternationalRugbyTeams"
            assert columns == ["TeamName", "Tier", "Hemisphere"]
            return pd.DataFrame(
                [
                    {"TeamName": "FRANCE", "Tier": 1, "Hemisphere": "NH"},
                    {"TeamName": "WALES", "Tier": 2, "Hemisphere": "NH"},
                ]
            )

    out = helpers.add_international_features(df, FakeSqlClient())
    assert len(out) == 2
    assert out.loc[0, "Home_Tier"] == 1
    assert out.loc[0, "Away_Tier"] == 2
    assert out.loc[0, "Diff_Tier"] == -1
    assert out.loc[0, "Home_Hemisphere"] == "NH"
    assert out.loc[0, "Away_Hemisphere"] == "NH"

    # Unknown team yields nullable tiers, but row is retained
    assert out.loc[1, "Home_Tier"] == 1
    assert pd.isna(out.loc[1, "Away_Tier"])

