from __future__ import annotations

from datetime import date, datetime, timedelta
import pandas as pd

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
        if expected_norm == "bool":
            return pd.api.types.is_bool_dtype(series)
        if expected_norm == "date":
            # either already datetime, or a string we can reasonably parse
            return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_string_dtype(series)

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
