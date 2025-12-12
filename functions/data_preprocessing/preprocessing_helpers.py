from typing import TYPE_CHECKING

from azure.storage.blob import BlobClient
from io import BytesIO
import pandas as pd

from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient
from functions.utils.utils import matches_type, drop_na_rows

if TYPE_CHECKING:
    from functions.data_preprocessing.preprocessing_services import PreprocessingEvent


settings = get_settings()
logger = get_logger(__name__)

### Transformation functions ###
def transform_kaggle_historical_data_to_international_results(source_data: pd.DataFrame, preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> pd.DataFrame:
    """
    Transform the Kaggle historical data to international results.
    Simple renaming and type conversions to match the target schema.
    """
    try:
        # if there are no rows, return an empty dataframe
        if source_data.empty:
            logger.warning(
                "No Kaggle historical rows for preprocessing_event=%s",
                preprocessing_event.id,
            )
            return pd.DataFrame()
        # transform the data from the source schema to the target schema
        df = source_data.copy()
        # drop duplicate rows
        df = df.drop_duplicates()
        # drop rows where any of the columns are NaN
        df = drop_na_rows(df, columns=["date", "home_team", "away_team", "home_score", "away_score", "competition","neutral", "world_cup"])
        transformed_data = pd.DataFrame(index=df.index)
        # rename the columns to the target schema
        transformed_data["MatchDate"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        transformed_data["HomeTeam"] = df["home_team"].str.upper()
        transformed_data["AwayTeam"] = df["away_team"].str.upper()
        transformed_data["HomeScore"] = df["home_score"].astype("Int64")
        transformed_data["AwayScore"] = df["away_score"].astype("Int64")
        transformed_data["CompetitionName"] = df["competition"].str.upper()
        transformed_data["Venue"] = df["stadium"].str.upper()
        transformed_data["City"] = df["city"].str.upper()
        transformed_data["Country"] = df["country"].str.upper()
        transformed_data["Neutral"] = df["neutral"].astype(bool)
        transformed_data["WorldCup"] = df["world_cup"].astype(bool)
        transformed_data = transformed_data.reset_index(drop=True)

        return transformed_data
    except Exception as e:
        logger.error("Error transforming Kaggle historical data to international results for preprocessing event %s: %s", preprocessing_event.id, e)
        raise

def transform_rugby365_results_data_to_international_results(source_data: pd.DataFrame, preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> pd.DataFrame:
    """
    Transform the Rugby365 results data to international results.
    Filter the data to only include the competitions in the list and add city, country, neutral, and world cup columns.
    Rename the columns to the target schema.
    """
    try:
        # get list of competitions
        competition_name_list = [
            "Rugby Championship",
            "Internationals",
            "Six Nations",
            "Rugby World Cup",
            "Pacific Nations Cup",
            "Rugby Europe Championship",
        ]
        # filter the data to only include the competitions in the list
        df = source_data[source_data["competition_name"].isin(competition_name_list)].copy()
        # drop duplicate rows
        df = df.drop_duplicates()
        # drop rows where any of the columns are NaN
        df = drop_na_rows(df, columns=["match_date", "home_team", "away_team", "home_score", "away_score", "competition_name"])
        # if there are no rows, return an empty dataframe
        if df.empty:
            logger.warning(
                "No Rugby365 rows matched competition filter for preprocessing_event=%s",
                preprocessing_event.id,
            )
            return pd.DataFrame()
        # determine city and country from the venue and add to the dataframe
        df = add_city_and_country_to_dataframe(df, sql_client, venue_column="venue", city_column="city", country_column="country")
        # determine if the match is a neutral match and add to the dataframe
        df = determine_neutral_match_international(df, home_team_column="home_team", away_team_column="away_team", country_column="country", neutral_column="neutral")
        # determine if the match is a world cup match and add to the dataframe
        df = determine_world_cup_match(df, world_cup_column="world_cup", competition_name_column="competition_name")
        # create a new dataframe to store the transformed data
        transformed_data = pd.DataFrame(index=df.index)
        # Keep MatchDate as a pandas datetime64 (normalised to date)
        transformed_data["MatchDate"] = pd.to_datetime(df["match_date"], errors="coerce").dt.normalize()
        transformed_data["HomeTeam"] = df["home_team"].str.upper()
        transformed_data["AwayTeam"] = df["away_team"].str.upper()
        transformed_data["HomeScore"] = df["home_score"].astype("Int64")
        transformed_data["AwayScore"] = df["away_score"].astype("Int64")
        transformed_data["CompetitionName"] = df["competition_name"].str.upper()
        transformed_data["Venue"] = df["venue"].str.upper()
        transformed_data["City"] = df["city"].str.upper()
        transformed_data["Country"] = df["country"].str.upper()
        transformed_data["Neutral"] = df["neutral"].astype(bool)
        transformed_data["WorldCup"] = df["world_cup"].astype(bool)

        transformed_data = transformed_data.reset_index(drop=True)
        return transformed_data
    except Exception as e:
        logger.error("Error transforming Rugby365 results data to international results for preprocessing event %s: %s", preprocessing_event.id, e)
        raise

def transform_rugby365_fixtures_data_to_international_fixtures(source_data: pd.DataFrame, preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> pd.DataFrame:
    """
    Transform the Rugby365 fixtures data to international fixtures.
    Filter the data to only include the competitions in the list and add city, country, neutral, and world cup columns.
    Rename the columns to the target schema.
    """
    try:
        # get list of competitions
        competition_name_list = [
            "Rugby Championship",
            "Internationals",
            "Six Nations",
            "Rugby World Cup",
            "Pacific Nations Cup",
            "Rugby Europe Championship",
        ]
        # filter the data to only include the competitions in the list
        df = source_data[source_data["competition_name"].isin(competition_name_list)].copy()
        # drop duplicate rows
        df = df.drop_duplicates()
        # drop rows where any of the columns are NaN
        df = drop_na_rows(df, columns=["kickoff_time_local", "match_date", "home_team", "away_team", "competition_name"])
        # there are no rows that match the competition filter
        if df.empty:
            logger.warning(
                "No Rugby365 fixtures rows matched competition filter for preprocessing_event=%s",
                preprocessing_event.id,
            )
            return pd.DataFrame()
        # parse the kickoff time and add to the dataframe
        df = add_kickoff_datetime_from_date_and_time(df, date_column="match_date", time_column="kickoff_time_local", target_column="datetime_kickoff_time")
        # determine city and country from the venue and add to the dataframe
        df = add_city_and_country_to_dataframe(df, sql_client, venue_column="venue", city_column="city", country_column="country")
        # determine if the match is a neutral match and add to the dataframe
        df = determine_neutral_match_international(df, home_team_column="home_team", away_team_column="away_team", country_column="country", neutral_column="neutral")
        # determine if the match is a world cup match and add to the dataframe
        df = determine_world_cup_match(df, world_cup_column="world_cup", competition_name_column="competition_name")
        # create a new dataframe to store the transformed data
        transformed_data = pd.DataFrame(index=df.index)
        # rename the columns to the target schema
        transformed_data["KickoffTimeLocal"] = df["datetime_kickoff_time"]
        transformed_data["MatchDate"] = pd.to_datetime(df["match_date"], errors="coerce").dt.normalize()
        transformed_data["HomeTeam"] = df["home_team"].str.upper()
        transformed_data["AwayTeam"] = df["away_team"].str.upper()
        transformed_data["CompetitionName"] = df["competition_name"].str.upper()
        transformed_data["Venue"] = df["venue"].str.upper()
        transformed_data["City"] = df["city"].str.upper()
        transformed_data["Country"] = df["country"].str.upper()
        transformed_data["Neutral"] = df["neutral"].astype(bool)
        transformed_data["WorldCup"] = df["world_cup"].astype(bool)
        transformed_data = transformed_data.reset_index(drop=True)
        return transformed_data
    except Exception as e:
        logger.error("Error transforming Rugby365 fixtures data to international fixtures for preprocessing event %s: %s", preprocessing_event.id, e)
        raise

### Validation functions ###

def validate_transformed_data(transformed_data: pd.DataFrame, target_schema: dict) -> None:
    """
    Validate the transformed data against the target schema.
    """
    try:
        # check if the transformed data has any rows/columns
        if transformed_data.empty:
            logger.error("Transformed data is empty")
            raise ValueError("Transformed data is empty")

        required = target_schema.get("required", {})
        types = target_schema.get("data_types", {})

        # required columns
        for col in target_schema["columns"]:
            if required.get(col, False) and col not in transformed_data.columns:
                logger.error("Transformed data is missing required column: %s", col)
                raise ValueError(f"Transformed data is missing required column: {col}")

        # type checks (only for columns that exist)
        for col in target_schema["columns"]:
            if col in transformed_data.columns:
                expected_type = types.get(col)
                if expected_type and not matches_type(transformed_data[col], expected_type):
                    logger.error(
                        "Transformed data has unexpected data type for column: %s. Expected=%s, got=%s",
                        col,
                        expected_type,
                        transformed_data[col].dtype,
                    )
                    raise ValueError(f"Transformed data has unexpected data type for column: {col}. Expected={expected_type}, got={transformed_data[col].dtype}")
        return 
    except Exception as e:
        logger.error("Error validating transformed data: %s", e)
        raise

def validate_source_data(source_data: pd.DataFrame, source_schema: dict) -> None:
    """
    Validate the source data against the source schema.
    """
    try:
        # check if the source data has any rows/columns
        if source_data.empty:
            logger.error("Source data is empty")
            raise ValueError("Source data is empty")

        required = source_schema.get("required", {})
        types = source_schema.get("data_types", {})

        # required columns
        for col in source_schema["columns"]:
            if required.get(col, False) and col not in source_data.columns:
                logger.error("Source data is missing required column: %s", col)
                raise ValueError(f"Source data is missing required column: {col}")

        # type checks (only for columns that exist)
        for col in source_schema["columns"]:
            if col in source_data.columns:
                expected_type = types.get(col)
                if expected_type and not matches_type(source_data[col], expected_type):
                    logger.error(
                        "Source data has unexpected data type for column: %s. Expected=%s, got=%s",
                        col,
                        expected_type,
                        source_data[col].dtype,
                    )
                    raise ValueError(f"Source data has unexpected data type for column: {col}. Expected={expected_type}, got={source_data[col].dtype}")
        return
    except Exception as e:
        logger.error("Error validating source data: %s", e)
        raise

### Helper functions ###
def get_source_data(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> pd.DataFrame:
    """
    Get the source data for a preprocessing event.
    """
    try:
        blob_client = BlobClient.from_connection_string(
                conn_str=settings.storage_connection,
                container_name=preprocessing_event.container_name,
                blob_name=preprocessing_event.blob_path,
            )
        data_stream = blob_client.download_blob()
        raw_bytes = data_stream.readall()
        data_frame = pd.read_csv(BytesIO(raw_bytes))
        return data_frame
    except Exception as e:
        logger.error("Error getting source data for preprocessing event %s: %s", preprocessing_event.id, e)
        raise

def get_source_schema(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> dict:
    """
    Get the expected source schema for a preprocessing event.
    """
    try:
        # get the logical source schema
        schema = sql_client.get_schema(
            integration_provider=preprocessing_event.integration_provider,
            integration_type=preprocessing_event.integration_type,
        )
        # convert the schema to a richer dictionary keyed by column name
        schema_dict = {
            "columns": [col["column_name"] for col in schema],
            "data_types": {col["column_name"]: col["data_type"] for col in schema},
            "required": {col["column_name"]: bool(col["is_required"]) for col in schema},
        }
        return schema_dict
    except Exception as e:
        logger.error("Error getting source schema for preprocessing event %s: %s", preprocessing_event.id, e)
        raise

def get_target_schema(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> dict:
    """
    Get the target schema for a preprocessing event.
    """
    try:
        # get the logical target table schema
        schema = sql_client.get_schema(table_name=preprocessing_event.target_table)
        # convert the schema to a richer dictionary keyed by column name
        schema_dict = {
            "columns": [col["column_name"] for col in schema],
            "data_types": {col["column_name"]: col["data_type"] for col in schema},
            "required": {col["column_name"]: bool(col["is_required"]) for col in schema},
        }
        return schema_dict
    except Exception as e:
        logger.error("Error getting target schema for preprocessing event %s: %s", preprocessing_event.id, e)
        raise


def write_data_to_target_table(
    transformed_data: pd.DataFrame,
    preprocessing_event: "PreprocessingEvent",
    sql_client: SqlClient,
) -> None:
    """
    Wrapper function that calls the sql client to write the data to the target table.
    """
    try:
        sql_client.write_dataframe_to_table(
            df=transformed_data,
            table_name=preprocessing_event.target_table,
        )
    except Exception as e:
        logger.error(
            "Error writing data to target table %s for preprocessing event %s: %s",
            preprocessing_event.target_table,
            preprocessing_event.id,
            e,
        )
        raise

### Transform helpers ###
def add_kickoff_datetime_from_date_and_time(
    df: pd.DataFrame,
    date_column: str,
    time_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Combine a date column and a time-only column into a full datetime.

    Expected inputs:
      - date_column: date or date-like string (e.g. '2025-03-22')
      - time_column: time string in 12-hour format (e.g. '4:05am')

    The resulting datetime uses the match date with the supplied local time.
    """
    try:
        df = df.copy()

        match_date = pd.to_datetime(
            df[date_column],
            errors="coerce",
        ).dt.normalize()

        kickoff_time = pd.to_datetime(
            df[time_column],
            format="%I:%M%p",
            errors="coerce",
        )

        df[target_column] = (
            match_date
            + pd.to_timedelta(kickoff_time.dt.hour, unit="h")
            + pd.to_timedelta(kickoff_time.dt.minute, unit="m")
        )

        return df

    except Exception as e:
        logger.error(
            "Error combining date (%s) and time (%s) into datetime: %s",
            date_column,
            time_column,
            e,
        )
        raise

def add_city_and_country_to_dataframe(
    df: pd.DataFrame,
    sql_client: SqlClient,
    venue_column: str,
    city_column: str,
    country_column: str,
) -> pd.DataFrame:
    """
    Add the city and country to the dataframe based on the venue.
    The venue database is a dictionary of venue names and their corresponding city and country.
    """
    try:
        venue_db = sql_client.get_venue_database()

        # Normalise venue names to uppercase for case-insensitive matching
        venue_db = venue_db.copy()
        venue_db["_venue_key"] = venue_db[venue_column].str.upper().str.strip()

        venue_lookup = venue_db.set_index("_venue_key")

        df = df.copy()
        venue_key = df[venue_column].str.upper().str.strip()

        df[city_column] = venue_key.map(venue_lookup[city_column])
        df[country_column] = venue_key.map(venue_lookup[country_column])

        return df

    except Exception as e:
        logger.error("Error adding city and country to dataframe: %s", e)
        raise

def determine_neutral_match_international(
    df: pd.DataFrame,
    home_team_column: str,
    away_team_column: str,
    country_column: str,
    neutral_column: str,
) -> pd.DataFrame:
    """
    Mark matches as neutral where the venue country is different from both the
    home and away team names.

    If the country is unknown (NaN), the match is treated as non-neutral.
    """
    try:
        df = df.copy()

        country_values = df[country_column].astype(str).str.strip().str.upper()
        home_team_values = df[home_team_column].astype(str).str.strip().str.upper()
        away_team_values = df[away_team_column].astype(str).str.strip().str.upper()

        known_country = df[country_column].notna()

        df[neutral_column] = (
            known_country
            & (country_values != home_team_values)
            & (country_values != away_team_values)
        )

        return df

    except Exception as e:
        logger.error("Error determining neutral match international: %s", e)
        raise


def determine_world_cup_match(
    df: pd.DataFrame,
    world_cup_column: str,
    competition_name_column: str,
) -> pd.DataFrame:
    """
    Mark matches as World Cup matches where the competition name contains
    'World Cup' (case-insensitive).
    """
    try:
        df = df.copy()
        df[world_cup_column] = df[competition_name_column].astype(str).str.contains(
            "world cup",
            case=False,
            na=False,
        )
        return df

    except Exception as e:
        logger.error("Error determining world cup match: %s", e)
        raise


