from typing import TYPE_CHECKING

from azure.storage.blob import BlobClient
from io import BytesIO
import pandas as pd
import numpy as np

from functions.config.settings import get_settings
from functions.logging.logger import get_logger
from functions.sql.sql_client import SqlClient
from functions.utils.utils import matches_type, drop_na_rows, time_decay_weight, roll_mean, roll_sum

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

## Model ready transformation functions ###
def transform_international_results_to_model_ready_data(
    source_data: pd.DataFrame,
    preprocessing_event: "PreprocessingEvent",    
    sql_client: SqlClient,
) -> pd.DataFrame:
    """
    Transform the international results data to model-ready data.
    """
    try:
        results_df = source_data.copy()
        if results_df.empty:
            logger.warning("No international results rows for preprocessing_event=%s", preprocessing_event.id)
            return pd.DataFrame()

        # Drop audit columns
        audit_cols = ["CreatedAt"]
        results_df = results_df.drop(columns=[c for c in audit_cols if c in results_df.columns], errors="ignore")

        # Filter to canonical teams/venues
        results_df = team_name_standardization(
            results_df, sql_client,
            table_name="dbo.InternationalRugbyTeams",
            home_team_col="HomeTeam", away_team_col="AwayTeam",
        )
        results_df = venue_standardisation(
            results_df, sql_client,
            table_name="dbo.RugbyVenues",
            venue_col="Venue",
        )

        # Add static international team features (Tier, Hemisphere, etc.)
        results_df = add_international_features(results_df, sql_client)

        # Long format staging
        team_long = to_team_match_long_format(
            results_df,
            id_col="ID",
            date_col="MatchDate",
            home_team_col="HomeTeam",
            away_team_col="AwayTeam",
            home_score_col="HomeScore",
            away_score_col="AwayScore",
            extra_context_cols=("CompetitionName", "Venue", "City", "Country", "Neutral", "WorldCup"),
        )

        # Rolling team form
        team_form_long = compute_rolling_team_form(team_long, form_window=10)

        # Attach features back to matches (correct arg order)
        model_base_df = attach_rolling_team_form_to_matches(results_df, team_form_long)

        # Targets (drop null scores first)
        model_base_df = model_base_df.dropna(subset=["HomeScore", "AwayScore"])
        # compute the time decay weight
        model_base_df["TimeDecayWeight"] = time_decay_weight(
            model_base_df,
            date_col="MatchDate",
            half_life_years=10.0,
        )
        # add the match targets
        model_base_df = add_match_targets(
            model_base_df,
            home_score_col="HomeScore",
            away_score_col="AwayScore",
            drop_draws=True,
        )

        # Don’t let the model see scores
        model_base_df = model_base_df.drop(columns=["HomeScore", "AwayScore"])
        # make sure columns are correct data types
        model_base_df["MatchDate"] = pd.to_datetime(
            model_base_df["MatchDate"],
            errors="coerce"
        ).dt.normalize()
        return model_base_df

    except Exception as e:
        logger.error("Error transforming international results to model ready data: %s", e)
        raise

def transform_international_fixtures_to_model_ready_data(
    source_data: pd.DataFrame,
    preprocessing_event: "PreprocessingEvent",    
    sql_client: SqlClient,
) -> pd.DataFrame:
    """
    Transform the international fixtures data to model-ready data.
    """
    try:
        fixtures_df = source_data.copy()
        if fixtures_df.empty:
            logger.warning("No international fixtures rows for preprocessing_event=%s", preprocessing_event.id)
            return pd.DataFrame()
        
        # Add the historical results to the fixtures to allow form computation
        results_df = sql_client.read_table_to_dataframe(
                table_name="dbo.InternationalMatchResults",
                where_sql=None,
                params= None,
            )
        
        fixtures_df = pd.concat([fixtures_df, results_df], ignore_index=True)

        # Drop audit columns
        audit_cols = ["CreatedAt"]
        fixtures_df = fixtures_df.drop(columns=[c for c in audit_cols if c in fixtures_df.columns], errors="ignore")

        # Filter to canonical teams/venues
        fixtures_df = team_name_standardization(
            fixtures_df, sql_client,
            table_name="dbo.InternationalRugbyTeams",
            home_team_col="HomeTeam", away_team_col="AwayTeam",
        )
        fixtures_df = venue_standardisation(
            fixtures_df, sql_client,
            table_name="dbo.RugbyVenues",
            venue_col="Venue",
        )

        # Add static international team features (Tier, Hemisphere, etc.)
        fixtures_df = add_international_features(fixtures_df, sql_client)

        # Long format staging
        team_long = to_team_match_long_format(
            fixtures_df,
            id_col="ID",
            date_col="MatchDate",
            home_team_col="HomeTeam",
            away_team_col="AwayTeam",
            home_score_col="HomeScore",
            away_score_col="AwayScore",
            # Include KickoffTimeLocal so we can deterministically order same-day fixtures
            # after historical results when computing rolling form.
            extra_context_cols=("KickoffTimeLocal", "CompetitionName", "Venue", "City", "Country", "Neutral", "WorldCup"),
        )

        # Rolling team form
        team_form_long = compute_rolling_team_form(team_long, form_window=10)

        # Attach features back to matches (correct arg order)
        model_base_df = attach_rolling_team_form_to_matches(fixtures_df, team_form_long)

        # Compute the time decay weight (kept for parity with results model data).
        model_base_df["TimeDecayWeight"] = time_decay_weight(
            model_base_df,
            date_col="MatchDate",
            half_life_years=10.0,
        )

        # Don’t let the model see scores
        model_base_df = model_base_df.drop(columns=["HomeScore", "AwayScore"])
        # make sure columns are correct data types
        model_base_df["MatchDate"] = pd.to_datetime(
            model_base_df["MatchDate"],
            errors="coerce"
        ).dt.normalize()

        # only return the fixtures rows
        model_base_df = model_base_df[model_base_df["KickoffTimeLocal"].notna()].reset_index(drop=True)
        return model_base_df

    except Exception as e:
        logger.error("Error transforming international fixtures to model ready data: %s", e)
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
        # Stage 2 support: when a source_table is provided, read from SQL instead of blob.
        # This enables "preprocessed table -> model table" pipelines.
        source_table = getattr(preprocessing_event, "source_table", None)
        if source_table:
            # Optional filtering: if the source table contains batch/system_event ids, filter to this event.
            schema_rows = sql_client.get_schema(table_name=source_table)
            colnames = [c.get("column_name") for c in schema_rows if c.get("column_name")]
            lower_to_actual = {c.lower(): c for c in colnames}

            where_parts: list[str] = []
            params: dict[str, object] = {}

            batch_col = lower_to_actual.get("batch_id")
            if batch_col:
                where_parts.append(f"{batch_col} = :batch_id")
                params["batch_id"] = str(preprocessing_event.batch_id)

            sys_col = lower_to_actual.get("system_event_id")
            if sys_col:
                where_parts.append(f"{sys_col} = :system_event_id")
                params["system_event_id"] = str(preprocessing_event.system_event_id)

            where_sql = " AND ".join(where_parts) if where_parts else None
            dataframe = sql_client.read_table_to_dataframe(
                table_name=source_table,
                where_sql=where_sql,
                params=params or None,
            )
        else:
            # Default behaviour (stage 1): read the ingested blob.
            blob_client = BlobClient.from_connection_string(
                conn_str=settings.storage_connection,
                container_name=preprocessing_event.container_name,
                blob_name=preprocessing_event.blob_path,
            )
            data_stream = blob_client.download_blob()
            raw_bytes = data_stream.readall()
            dataframe = pd.read_csv(BytesIO(raw_bytes))
        return dataframe
    except Exception as e:
        logger.error("Error getting source data for preprocessing event %s: %s", preprocessing_event.id, e)
        raise

def get_source_schema(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> dict:
    """
    Get the expected source schema for a preprocessing event.
    """
    try:
        # If a source_table is provided, use its schema directly (stage 2 SQL->SQL pipelines).
        # Otherwise, fall back to integration_provider/type lookups (stage 1 blob->SQL pipelines).
        source_table = getattr(preprocessing_event, "source_table", None)
        if source_table:
            schema = sql_client.get_schema(table_name=source_table)
        else:
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

def truncate_target_table(preprocessing_event: "PreprocessingEvent", sql_client: SqlClient) -> None:
    """
    Truncate the target table for a preprocessing event.
    """
    try:
        sql_client.truncate_table(table_name=preprocessing_event.target_table)
    except Exception as e:
        logger.error("Error truncating target table for preprocessing event %s: %s", preprocessing_event.id, e)
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


#### model ready data helpers ###
def team_name_standardization(
    df: pd.DataFrame,
    sql_client: SqlClient,
    table_name: str = "dbo.InternationalRugbyTeams",
    home_team_col: str = "HomeTeam",
    away_team_col: str = "AwayTeam",
) -> pd.DataFrame:
    """
    Current logic: Drop matches where either HomeTeam or AwayTeam is not present
    in the canonical teams table.

    Future logic: Apply alias mapping / standardisation before filtering.

    Accepts:
    - df: pd.DataFrame - the dataframe to standardize the team names for
    - sql_client: SqlClient - the sql client to use to get the canonical teams table
    - table_name: str - the name of the table to get the canonical teams table from
    - home_team_col: str - the name of the column to use for the home team
    - away_team_col: str - the name of the column to use for the away team

    Returns:
    - pd.DataFrame - the dataframe with the team names standardized
    """
    try:
        df = df.copy()
        # drop rows where the home or away team is null
        df = df.dropna(subset=[home_team_col, away_team_col])
        # get the canonical teams table from the sql client
        teams_df = sql_client.read_table_to_dataframe(table_name=table_name, columns=["TeamName"])
        if teams_df.empty or "TeamName" not in teams_df.columns:
            raise ValueError(f"Canonical teams table {table_name} is empty or missing TeamName column")

        valid_teams = set(
            teams_df["TeamName"].astype(str).str.upper().str.strip()
        )

        # Normalise just for comparison (not changing stored values)
        home = df[home_team_col].astype(str).str.upper().str.strip()
        away = df[away_team_col].astype(str).str.upper().str.strip()
        # filter the dataframe to only include matches where the home and away team names are in the canonical teams table
        mask_valid = home.isin(valid_teams) & away.isin(valid_teams)

        # Log what got dropped (before filtering)
        dropped_home = set(home[~home.isin(valid_teams)])
        dropped_away = set(away[~away.isin(valid_teams)])
        dropped_team_names = sorted(dropped_home | dropped_away)

        if dropped_team_names:
            logger.warning("Dropping matches with unknown teams: %s", dropped_team_names)

        filtered_df = df.loc[mask_valid].reset_index(drop=True)
        return filtered_df

    except Exception as e:
        logger.error("Error filtering/standardising team names: %s", e)
        raise

def venue_standardisation(
    df: pd.DataFrame,
    sql_client: SqlClient,
    table_name: str,
    venue_col: str = "Venue",
) -> pd.DataFrame:
    """
    Current logic:
      - Drop matches where Venue is not present in the canonical venues table.

    Future logic:
      - Apply alias mapping before filtering.
    """
    try:
        df = df.copy()
        # drop rows where the venue is null
        df = df.dropna(subset=[venue_col])
        # Load canonical venues
        venues_df = sql_client.read_table_to_dataframe(table_name=table_name,columns=["VenueName"])

        if venues_df.empty or "VenueName" not in venues_df.columns:
            raise ValueError(
                f"Canonical venues table {table_name} is empty or missing VenueName column"
            )

        valid_venues = set(
            venues_df["VenueName"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # Normalise for comparison only
        venue_norm = (
            df[venue_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        mask_valid = venue_norm.isin(valid_venues)

        # Log dropped venues (before filtering)
        dropped_venues = sorted(set(venue_norm[~mask_valid]))

        if dropped_venues:
            logger.warning(
                "Dropping %s matches due to unknown venues: %s",
                len(dropped_venues),
                dropped_venues,
            )

        filtered_df = df.loc[mask_valid].reset_index(drop=True)
        return filtered_df

    except Exception as e:
        logger.error("Error filtering venues: %s", e)
        raise

def add_match_targets(
    df: pd.DataFrame,
    home_score_col: str = "HomeScore",
    away_score_col: str = "AwayScore",
    drop_draws: bool = True,
) -> pd.DataFrame:
    """
    Add target columns derived from final match scores.

    Targets:
      - HomeWin: binary (1 = home win, 0 = home loss)
      - PointDiff: HomeScore - AwayScore

    Accepts:
    - df: pd.DataFrame - the dataframe to add the match targets to
    - home_score_col: str - the name of the column to use for the home score
    - away_score_col: str - the name of the column to use for the away score
    - drop_draws: bool - whether to drop draws for binary classification

    Returns:
    - pd.DataFrame - the dataframe with the match targets added
    """
    try:
        df = df.copy()
        # Point differential
        df["PointDiff"] = df[home_score_col] - df[away_score_col]
        # Binary outcome
        df["HomeWin"] = (df["PointDiff"] > 0).astype(int)
        # Optional: remove draws for binary classification
        if drop_draws:
            df = df[df["PointDiff"] != 0].reset_index(drop=True)

        return df
    except Exception as e:
        logger.error("Error adding match targets: %s", e)
        raise

def to_team_match_long_format(
    matches: pd.DataFrame,
    id_col: str = "ID",
    date_col: str = "MatchDate",
    home_team_col: str = "HomeTeam",
    away_team_col: str = "AwayTeam",
    home_score_col: str = "HomeScore",
    away_score_col: str = "AwayScore",
    extra_context_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Convert wide match rows into long team-match rows (2 rows per match):
      - one from the HomeTeam perspective
      - one from the AwayTeam perspective

    Output columns are designed for rolling form calculations.

    Accepts:
    - matches: pd.DataFrame - the dataframe to convert to a long format
    - id_col: str - the name of the column to use for the id
    - date_col: str - the name of the column to use for the date
    - home_team_col: str - the name of the column to use for the home team
    - away_team_col: str - the name of the column to use for the away team
    - home_score_col: str - the name of the column to use for the home score
    - away_score_col: str - the name of the column to use for the away score
    - extra_context_cols: tuple[str, ...] - the name of the columns to use for the extra context

    Returns:
    - pd.DataFrame - the dataframe with the matches converted to a long format
    """
    
    try:
        df = matches.copy()
        # Keep only context columns that actually exist (so this function is resilient)
        context_cols = [c for c in extra_context_cols if c in df.columns]
        # Home perspective
        home_long = df[[id_col, date_col, home_team_col, away_team_col, home_score_col, away_score_col, *context_cols]].copy()
        home_long.rename(columns={
            home_team_col: "Team",
            away_team_col: "Opponent",
            home_score_col: "PointsFor",
            away_score_col: "PointsAgainst",
        }, inplace=True)
        home_long["IsHome"] = 1

        # Away perspective
        away_long = df[[id_col, date_col, home_team_col, away_team_col, home_score_col, away_score_col, *context_cols]].copy()
        away_long.rename(columns={
            away_team_col: "Team",
            home_team_col: "Opponent",
            away_score_col: "PointsFor",
            home_score_col: "PointsAgainst",
        }, inplace=True)
        away_long["IsHome"] = 0

        # join the home and away long dataframes
        long_df = pd.concat([home_long, away_long], ignore_index=True)

        # Outcome computations from the team perspective.
        long_df["PointDiff"] = long_df["PointsFor"] - long_df["PointsAgainst"]
        has_scores = long_df["PointsFor"].notna() & long_df["PointsAgainst"].notna()

        long_df["Win"] = pd.Series(pd.NA, index=long_df.index, dtype="Int64")
        long_df.loc[has_scores, "Win"] = (long_df.loc[has_scores, "PointDiff"] > 0).astype("Int64")

        long_df["Draw"] = pd.Series(pd.NA, index=long_df.index, dtype="Int64")
        long_df.loc[has_scores, "Draw"] = (long_df.loc[has_scores, "PointDiff"] == 0).astype("Int64")

        long_df["Loss"] = pd.Series(pd.NA, index=long_df.index, dtype="Int64")
        long_df.loc[has_scores, "Loss"] = (long_df.loc[has_scores, "PointDiff"] < 0).astype("Int64")

        # Sort for stable rolling calculations
        long_df[date_col] = pd.to_datetime(long_df[date_col], errors="coerce")

        # Deterministic ordering:
        if "KickoffTimeLocal" in long_df.columns:
            kickoff_dt = pd.to_datetime(long_df["KickoffTimeLocal"], errors="coerce")
            long_df["_SortDateTime"] = kickoff_dt.fillna(long_df[date_col])
            sort_cols = ["Team", "_SortDateTime", id_col]
        else:
            sort_cols = ["Team", date_col, id_col]

        long_df = long_df.sort_values(sort_cols).reset_index(drop=True)
        if "_SortDateTime" in long_df.columns:
            long_df = long_df.drop(columns=["_SortDateTime"])

        return long_df
    except Exception as e:
        logger.error("Error converting matches to team match long format: %s", e)
        raise

def compute_rolling_team_form(
    team_long: pd.DataFrame,
    form_window: int = 10,
    date_col: str = "MatchDate",
) -> pd.DataFrame:
    """
    Compute rolling team-form features per team using only prior matches.
    Accepts:
    - team_long: pd.DataFrame - the dataframe to compute the rolling team form for
    - form_window: int - the window size for the rolling team form
    - date_col: str - the name of the column to use for the date

    Returns:
    - pd.DataFrame - the dataframe with the rolling team form computed
    """
    try:
        df = team_long.copy()

        # Ensure correct ordering
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(["Team", date_col, "ID"]).reset_index(drop=True)

        # Compute per-team rolling features (transform keeps index aligned).
        # We rely on NaNs (for fixtures/no-scores rows) to be ignored by rolling mean/sum.
        df["FormWinRate"] = df.groupby("Team")["Win"].transform(lambda s: roll_mean(s, form_window))
        df["FormDrawRate"] = df.groupby("Team")["Draw"].transform(lambda s: roll_mean(s, form_window))
        df["FormPointsFor"] = df.groupby("Team")["PointsFor"].transform(lambda s: roll_mean(s, form_window))
        df["FormPointsAgainst"] = df.groupby("Team")["PointsAgainst"].transform(lambda s: roll_mean(s, form_window))
        df["FormPointDiff"] = df.groupby("Team")["PointDiff"].transform(lambda s: roll_mean(s, form_window))

        # FormGames = rolling count of *played* prior matches (ignore fixtures).
        played = (df["PointsFor"].notna() & df["PointsAgainst"].notna()).astype("Int64")
        df["FormGames"] = played.groupby(df["Team"]).transform(lambda s: roll_sum(s, form_window))
        # Rolling ops + shift introduce NaNs, which forces float dtype; cast back to nullable int.
        # Values should be whole numbers (counts), so rounding is safe.
        df["FormGames"] = df["FormGames"].round(0).astype("Int64")

        form_complete_df = df
        return form_complete_df

    except Exception as e:
        logger.error("Error computing rolling team form: %s", e)
        raise

def attach_rolling_team_form_to_matches(
    results_df: pd.DataFrame,
    team_form_long: pd.DataFrame,
    id_col: str = "ID",
    home_team_col: str = "HomeTeam",
    away_team_col: str = "AwayTeam",
) -> pd.DataFrame:
    """
    Attach pre-match rolling form features to each match row:
      - Home_* features (from the home team perspective)
      - Away_* features (from the away team perspective)
      - Diff_* features (Home - Away)
    """
    df = results_df.copy()

    rolling_cols = [
        "FormWinRate",
        "FormDrawRate",
        "FormPointsFor",
        "FormPointsAgainst",
        "FormPointDiff",
        "FormGames",
    ]

    # Home perspective rows (IsHome=1)
    home_feats = (
        team_form_long[team_form_long["IsHome"] == 1][[id_col, "Team", *rolling_cols]]
        .copy()
        .rename(columns={c: f"Home_{c}" for c in rolling_cols})
    )

    # Away perspective rows (IsHome=0)
    away_feats = (
        team_form_long[team_form_long["IsHome"] == 0][[id_col, "Team", *rolling_cols]]
        .copy()
        .rename(columns={c: f"Away_{c}" for c in rolling_cols})
    )

    # Join on match id + team name to avoid mismatches
    df = df.merge(home_feats, left_on=[id_col, home_team_col], right_on=[id_col, "Team"], how="left").drop(columns=["Team"])
    df = df.merge(away_feats, left_on=[id_col, away_team_col], right_on=[id_col, "Team"], how="left").drop(columns=["Team"])

    # Diff features (Home - Away) using the actual column names that exist
    df["Diff_FormWinRate"] = df["Home_FormWinRate"] - df["Away_FormWinRate"]
    df["Diff_FormPointDiff"] = df["Home_FormPointDiff"] - df["Away_FormPointDiff"]
    df["Diff_FormPointsFor"] = df["Home_FormPointsFor"] - df["Away_FormPointsFor"]
    df["Diff_FormPointsAgainst"] = df["Home_FormPointsAgainst"] - df["Away_FormPointsAgainst"]

    return df

def add_international_features(
    df: pd.DataFrame,
    sql_client: SqlClient,
) -> pd.DataFrame:
    """
    Add international team-level features (e.g. Tier, Hemisphere) for
    HomeTeam and AwayTeam.

    This is designed to be:
      - case/whitespace insensitive
      - non-destructive (nullable features; no row drops)
      - expects dbo.InternationalRugbyTeams to contain TeamName, Tier, Hemisphere
    """
    try:
        df = df.copy()

        teams = sql_client.read_table_to_dataframe(
            table_name="dbo.InternationalRugbyTeams",
            columns=["TeamName", "Tier", "Hemisphere"],
        )

        if teams.empty or "TeamName" not in teams.columns:
            logger.warning("add_international_features: dbo.InternationalRugbyTeams returned no rows; skipping.")
            return df

        teams = teams.copy()
        teams["_team_key"] = teams["TeamName"].astype(str).str.upper().str.strip()
        teams = teams.dropna(subset=["_team_key"]).drop_duplicates(subset=["_team_key"])

        # Build lookups
        tier_lookup = pd.to_numeric(teams.get("Tier"), errors="coerce")
        tier_lookup = pd.Series(tier_lookup.values, index=teams["_team_key"])

        hemisphere_lookup = None
        if "Hemisphere" in teams.columns:
            hemisphere_series = teams["Hemisphere"].astype(str).str.upper().str.strip()
            hemisphere_lookup = pd.Series(hemisphere_series.values, index=teams["_team_key"])

        home_key = df["HomeTeam"].astype(str).str.upper().str.strip()
        away_key = df["AwayTeam"].astype(str).str.upper().str.strip()

        df["Home_Tier"] = home_key.map(tier_lookup).astype("Int64")
        df["Away_Tier"] = away_key.map(tier_lookup).astype("Int64")
        df["Diff_Tier"] = (df["Home_Tier"] - df["Away_Tier"]).astype("Int64")

        if hemisphere_lookup is not None:
            df["Home_Hemisphere"] = home_key.map(hemisphere_lookup)
            df["Away_Hemisphere"] = away_key.map(hemisphere_lookup)

        return df
    except Exception as e:
        logger.error("Error adding international features: %s", e)
        raise




