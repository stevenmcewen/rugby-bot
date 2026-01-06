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
    category_ratio_threshold: float = 0.5,
    date_sample_size: int = 20,
) -> pd.DataFrame:
    """
    Normalize DataFrame dtypes after loading from sources like SQL/CSV so downstream
    code (ML, feature engineering, persistence) behaves predictably.

    Why this is necessary
    ---------------------
    Real-world connectors often produce “technically valid” but awkward pandas dtypes:

    - SQL DECIMAL frequently arrives as Python `decimal.Decimal` objects -> pandas uses
      dtype=object, which breaks many ML libraries and makes math/aggregations slower.
    - Numbers may arrive as strings (e.g., "12.34") depending on the source.
    - Text columns sometimes arrive as dtype=object; we prefer pandas' `string` dtype.
    - Low-cardinality label columns (team names, countries, enums) can be stored as
      `category` to reduce memory and speed up groupby/filter operations.

    What this function does
    -----------------------
    1) Converts to pandas extension dtypes where possible (convert_dtypes).
    2) For object/string columns, applies this decision order:
       a) If values are numeric-like -> convert to Float64
       b) If values are date-like -> convert to datetime64[ns]
       c) If low-cardinality labels -> convert to category
       d) Otherwise -> keep as pandas string dtype

    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if df.empty:
        return df

    out = df.copy().convert_dtypes()
    n_rows = len(out)

    # ---------- helper functions for normalization ----------
    def is_object_or_string(series: pd.Series) -> bool:
        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

    def is_decimalish(series: pd.Series) -> bool:
        # All non-nulls are numeric Python objects (Decimal/int/float)
        non_null = series.dropna()
        if non_null.empty:
            return False
        return non_null.map(lambda v: isinstance(v, (Decimal, int, float))).all()

    def is_dateish(series: pd.Series) -> bool:
        # Already datetime is date-ish.
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        non_null = series.dropna()
        if non_null.empty:
            return False

        sample = non_null.head(min(date_sample_size, len(non_null)))
        return sample.map(lambda v: isinstance(v, (date, datetime))).all()

    def can_parse_numeric(series: pd.Series) -> tuple[bool, pd.Series]:
        # Parse from string, see what fraction becomes numeric.
        s_str = series.astype("string").str.strip().replace({"": pd.NA})
        s_num = pd.to_numeric(s_str, errors="coerce")
        ratio = float(s_num.notna().mean())
        return (ratio >= numeric_parse_threshold and s_num.notna().any(), s_num)

    def is_low_cardinality(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        nunique = int(non_null.nunique())
        if nunique == 0 or nunique > max_categories:
            return False
        return (nunique / max(n_rows, 1)) <= category_ratio_threshold

    # ---------- main normalization loop ----------
    for col in out.columns:
        string = out[col]

        # Only normalize text/object-like columns. Numeric/datetime dtypes are already fine.
        if not is_object_or_string(string):
            continue

        # 1) SQL DECIMAL -> Float64
        if is_decimalish(string):
            out[col] = pd.to_numeric(string, errors="coerce").astype("Float64")
            continue

        # 2) Date-like python objects -> datetime64[ns]
        if is_dateish(string):
            out[col] = pd.to_datetime(string, errors="coerce")
            continue

        # 3) Numeric-looking strings -> Float64
        numeric_ok, s_num = can_parse_numeric(string)
        if numeric_ok:
            out[col] = s_num.astype("Float64")
            continue

        # 4) Low-cardinality label columns -> category
        if is_low_cardinality(string):
            out[col] = string.astype("category")
            continue

        # 5) Default: keep as pandas string dtype
        out[col] = string.astype("string").str.strip().replace({"": pd.NA})

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
