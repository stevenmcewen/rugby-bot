from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import pandas as pd
import hashlib
from decimal import Decimal
from typing import Any

from functions.logging.logger import get_logger

logger = get_logger(__name__)

def to_date_range(start_date: date, end_date: date) -> list[date]:
    """
    Create a list of dates from start_date to end_date (inclusive).

    Example:
        start_date = 2025-12-01
        end_date   = 2025-12-05

        -> [
            2025-12-01,
            2025-12-02,
            2025-12-03,
            2025-12-04,
            2025-12-05
        ]
    
    Accepts:
    - A start date
    - An end date
    Returns:
    - A list of dates from start_date to end_date (inclusive)
    Raises:
    - ValueError if start_date is greater than end_date
    """
    if start_date > end_date:
        raise ValueError("start_date must be less than or equal to end_date")
    num_days = (end_date - start_date).days
    date_range_list = [start_date + timedelta(days=i) for i in range(num_days + 1)]
    return date_range_list


def matches_type(series: pd.Series, expected: str) -> bool:
    """
    Check whether a pandas Series roughly matches an expected logical type.

    The expected types are semantic strings coming from our schema tables
    (e.g. "string", "int", "bool", "date").
    """
    try:
        if expected is None:
            return False

        expected_norm = str(expected).strip().lower()

        if expected_norm == "string":
            # Treat both dedicated string dtypes and object dtypes (typical
            # for CSV text columns) as valid "string" columns.
            return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        if expected_norm == "int":
            return pd.api.types.is_integer_dtype(series)
        if expected_norm in ("float", "double", "numeric", "decimal"):
            # For our validator, treat any numeric dtype as acceptable for "float/numeric"
            # because pandas may represent numeric columns as int or float depending on nulls.
            return pd.api.types.is_numeric_dtype(series)
        if expected_norm in ("bool", "bit"):
            if pd.api.types.is_bool_dtype(series):
                return True
            if pd.api.types.is_integer_dtype(series):
                vals = series.dropna().unique()
                return set(vals).issubset({0, 1})
            return False
        if expected_norm == "date":
            # Either already datetime, or a string/object we can reasonably parse later.
            return (
                pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series)
            )

        if expected_norm == "time":
            # SQL schema sometimes uses a semantic "time" for time-of-day or
            # kickoff timestamps. In pandas this often arrives as datetime64,
            # or as object/string when loaded from CSV/SQL.
            return (
                pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series)
            )

        if expected_norm == "datetime":
            # CSV-backed datetime columns typically arrive as object dtype.
            # Treat datetime64, string, and object columns as acceptable "datetime"
            # inputs (parsing/normalisation happens later in the pipeline).
            return (
                pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series)
            )

        # Fail closed: unknown expected types must not validate successfully.
        logger.warning("Unknown expected type for matches_type(): %r", expected)
        return False
    except Exception:
        return False

def drop_na_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop rows where any of the specified columns are NaN.
    Accepts:
    - A pandas DataFrame
    - A list of columns to check for NaN
    Returns:
    - A pandas DataFrame with the rows where any of the specified columns are NaN dropped
    Raises:
    - ValueError if the columns list is empty
    """
    if not columns:
        raise ValueError("columns list cannot be empty")
    try:
        df = df.copy()
        df = df.dropna(subset=columns)
        return df
    except Exception as e:
        logger.error("Error dropping NA rows: %s", e)
        raise


def stable_schema_hash(columns: list[str]) -> str:
    """
    Generate a stable hash of a list of columns (order-insensitive).
    Accepts:
    - A list of columns
    Returns:
    - A string representing the stable hash of the list of columns
    Raises:
    - ValueError if the columns list is empty
    """
    if not columns:
        raise ValueError("columns list cannot be empty")
    try:
        normalized = "|".join(sorted(columns))
        hash_value = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return hash_value
    except Exception as e:
        logger.error("Error generating stable schema hash: %s", e)
        raise 


def uct_now_iso() -> str:
    """
    Get the current UTC time in ISO 8601 format.
    Returns:
    - A string representing the current UTC time in ISO 8601 format
    """
    south_africa = datetime.now(timezone.utc)
    iso_time = south_africa.isoformat()

    return iso_time

def normalize_dataframe_dtypes(
    df: pd.DataFrame,
    *,
    numeric_parse_threshold: float = 0.95,
    max_categories: int = 50,
) -> pd.DataFrame:
    """
    Normalize a DataFrame's dtypes after loading from external sources (e.g. SQL).

    Why this exists:
    - SQL DECIMAL columns often arrive in Python as `decimal.Decimal`, which forces
      pandas to use dtype=object. Many ML libs expect numeric dtypes.
    - Some connectors also return numeric values as strings.

    Strategy (generic, model-agnostic):
    - Convert pandas extension dtypes where possible (`convert_dtypes`).
    - For object/string columns:
      - If values are mostly numeric-like, coerce to float.
      - Else if low-cardinality, coerce to `category`.
      - Else keep as pandas `string` dtype (not object).
    Accepts:
    - A pandas DataFrame
    - A numeric_parse_threshold
    - A max_categories
    Returns:
    - A pandas DataFrame with the dtypes normalized
    Raises:
    - ValueError if df is not a pandas DataFrame
    """
    # Check if the dataframe is valid
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # Check if the dataframe is empty
    if df.empty:
        return df

    # Copy the dataframe
    out = df.copy()
    # Convert the dtypes of the dataframe
    out = out.convert_dtypes()
    n_rows = len(out)

    # Iterate over the columns of the dataframe
    for col in out.columns:
        s = out[col]

        # Check if the column is an object or string dtype and if not, continue to the next column
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        # Fast path: SQL DECIMAL often arrives as Decimal objects -> cast to float.
        non_null = s.dropna()
        if not non_null.empty:
            # Check if every non-null is a Decimal (or int/float), if so, cast to float
            if non_null.map(lambda v: isinstance(v, (Decimal, int, float))).all():
                out[col] = pd.to_numeric(s, errors="coerce").astype("Float64")
                continue

        # Cast the column to a string dtype and strip the whitespace and replace empty strings with NA
        s_str = s.astype("string").str.strip().replace({"": pd.NA})
        s_num = pd.to_numeric(s_str, errors="coerce")

        non_na_ratio = float(s_num.notna().mean())
        if non_na_ratio >= numeric_parse_threshold and s_num.notna().any():
            out[col] = s_num.astype("Float64")
            continue

        # Low-cardinality -> category
        # (Use nunique on original values, not stringified, to preserve semantics.)
        nunique = int(non_null.nunique()) if not non_null.empty else 0
        if nunique > 0 and nunique <= max_categories and (nunique / max(n_rows, 1)) <= 0.5:
            out[col] = s.astype("category")
            continue

        # Otherwise keep as string dtype (avoid raw object)
        out[col] = s_str

    return out
    
def roll_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the rolling mean of a pandas Series.
    Accepts:
    - series: pd.Series - the series to compute the rolling mean for
    - window: int - the window size for the rolling mean
    Returns:
    - pd.Series - the rolling mean of the series
    """
    try:
        series = series.rolling(window, min_periods=1).mean().shift(1)
        return series
    except Exception as e:
        logger.error("Error computing rolling mean: %s", e)
        raise

def roll_sum(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the rolling sum of a pandas Series.
    Accepts:
    - series: pd.Series - the series to compute the rolling sum for
    - window: int - the window size for the rolling sum
    Returns:
    - pd.Series - the rolling sum of the series
    """
    try:
        series = series.rolling(window, min_periods=1).sum().shift(1)
        return series
    except Exception as e:
        logger.error("Error computing rolling sum: %s", e)
        raise

def time_decay_weight(
    df: pd.DataFrame,
    date_col: str,
    half_life_years: float = 10.0,
    reference_date: pd.Timestamp | None = None,
) -> pd.Series:
    """
    Compute exponential time-decay weights for a date column.

    Weight = 0.5 ** (age_years / half_life_years)

    Args:
        df: DataFrame containing the date column.
        date_col: Name of the date column in df.
        half_life_years: Years for weight to halve.
        reference_date: Optional "now" date; defaults to max(date_col).

    Returns:
        pd.Series of float weights in [0, 1], aligned to df.index.
    """
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.isna().any():
            bad = dates.isna().sum()
            raise ValueError(f"{bad} rows have invalid {date_col} values (could not parse datetime).")

        if reference_date is None:
            reference_date = dates.max()

        age_years = (reference_date - dates).dt.days / 365.25
        decay = 0.5 ** (age_years / half_life_years)
        return decay

    except Exception as e:
        logger.error("Error computing time decay weight: %s", e)
        raise
