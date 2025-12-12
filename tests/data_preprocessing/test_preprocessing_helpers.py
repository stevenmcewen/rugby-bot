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
    ):
        self.id = uuid4()
        self.container_name = container_name
        self.blob_path = blob_path
        self.integration_provider = integration_provider
        self.integration_type = integration_type
        self.target_table = target_table

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


