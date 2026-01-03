from datetime import date

import pandas as pd
import pytest
from decimal import Decimal

from functions.utils import utils

# Does the to_date_range function return a single date when the start and end are the same?
# must return a single date when the start and end are the same
def test_to_date_range_single_day():
    start = date(2025, 12, 8)
    end = date(2025, 12, 8)

    result = utils.to_date_range(start, end)

    assert result == [start]

# Does the to_date_range function return a list of dates when the start and end are different?
# must return a list of dates when the start and end are different
def test_to_date_range_multiple_days():
    start = date(2025, 12, 1)
    end = date(2025, 12, 5)

    result = utils.to_date_range(start, end)

    assert result == [
        date(2025, 12, 1),
        date(2025, 12, 2),
        date(2025, 12, 3),
        date(2025, 12, 4),
        date(2025, 12, 5),
    ]

# Does the to_date_range function raise a ValueError when the start is after the end?
# must raise a ValueError when the start is after the end
def test_to_date_range_start_after_end_raises_value_error():
    start = date(2025, 12, 10)
    end = date(2025, 12, 5)

    with pytest.raises(ValueError):
        utils.to_date_range(start, end)

# Does the matches_type function return False when the series is not of the expected type?
# must return False when the series is not of the expected type
def test_matches_type_unknown_expected_returns_false():
    s = pd.Series([1, 2, 3])
    assert utils.matches_type(s, "uuid") is False

# Does the matches_type function return False when the series is None?
# must return False when the series is None
def test_matches_type_none_expected_returns_false():
    s = pd.Series(["a", "b"])
    assert utils.matches_type(s, None) is False

# Does the matches_type function return True when the series is of the expected type (object and string)?
# must return True when the series is of the expected type
def test_matches_type_string_accepts_object_and_string_dtypes():
    s_obj = pd.Series(["a", "b"], dtype="object")
    s_str = pd.Series(["a", "b"], dtype="string")
    assert utils.matches_type(s_obj, "string") is True
    assert utils.matches_type(s_str, "string") is True

# Does the matches_type function return True when the series is of the expected type (int)?
# must return True when the series is of the expected type
def test_matches_type_int_accepts_integer_dtype():
    s = pd.Series([1, 2, 3], dtype="int64")
    assert utils.matches_type(s, "int") is True

# Does the matches_type function return True when the series is of the expected type (bool)?
# must return True when the series is of the expected type
def test_matches_type_bool_accepts_boolean_dtype():
    s = pd.Series([True, False, True], dtype="bool")
    assert utils.matches_type(s, "bool") is True

# Does the matches_type function return True when the series is of the expected type (date)?
# must return True when the series is of the expected type
def test_matches_type_date_accepts_datetime_and_strings():
    s_dt = pd.Series(pd.to_datetime(["2025-12-08", "2025-12-09"]))
    s_str = pd.Series(["2025-12-08", "2025-12-09"])
    assert utils.matches_type(s_dt, "date") is True
    assert utils.matches_type(s_str, "date") is True


def test_matches_type_datetime_accepts_datetime_and_object_and_strings():
    s_dt = pd.Series(pd.to_datetime(["2025-12-08 04:05:00", "2025-12-09 19:30:00"]))
    s_str = pd.Series(["2025-12-08 04:05:00", "2025-12-09 19:30:00"])
    s_obj = pd.Series(["2025-12-08 04:05:00", "2025-12-09 19:30:00"], dtype="object")
    assert utils.matches_type(s_dt, "datetime") is True
    assert utils.matches_type(s_str, "datetime") is True
    assert utils.matches_type(s_obj, "datetime") is True


def test_matches_type_time_accepts_datetime_and_object_and_strings():
    s_dt = pd.Series(pd.to_datetime(["2025-12-08 04:05:00", "2025-12-09 19:30:00"]))
    s_str = pd.Series(["04:05:00", "19:30:00"])
    s_obj = pd.Series(["2025-12-08 04:05:00", "2025-12-09 19:30:00"], dtype="object")
    assert utils.matches_type(s_dt, "time") is True
    assert utils.matches_type(s_str, "time") is True
    assert utils.matches_type(s_obj, "time") is True


# Does the drop_na_rows function drop rows with nan in any specified column?
# must drop rows with nan in any specified column
def test_drop_na_rows_drops_rows_with_nan_in_any_specified_column():
    df = pd.DataFrame(
        [
            {"a": 1, "b": 1},
            {"a": None, "b": 2},
            {"a": 3, "b": None},
            {"a": 4, "b": 4},
        ]
    )

    result = utils.drop_na_rows(df, columns=["a", "b"])

    assert list(result["a"]) == [1, 4]
    assert list(result["b"]) == [1, 4]

# Does the drop_na_rows function raise a ValueError when the columns are empty?
# must raise a ValueError when the columns are empty
def test_drop_na_rows_empty_columns_raises_value_error():
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(ValueError):
        utils.drop_na_rows(df, columns=[])

# Does the roll_mean function shift and use min_periods_1?
# must shift and use min_periods_1
def test_roll_mean_shifts_and_uses_min_periods_1():
    s = pd.Series([1.0, 2.0, 3.0])
    out = utils.roll_mean(s, window=2)
    # rolling(...).mean().shift(1) => first is NaN, then prior means
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 1.5


# Does the roll_sum function shift and use min_periods_1?
# must shift and use min_periods_1
def test_roll_sum_shifts_and_uses_min_periods_1():
    s = pd.Series([1, 2, 3])
    out = utils.roll_sum(s, window=2)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1
    assert out.iloc[2] == 3


# Does the time_decay_weight function use the reference date and half life and returns the expected weights?
# must use the reference date and half life and returns the expected weights
def test_time_decay_weight_uses_reference_date_and_half_life_and_returns_expected_weights():
    df = pd.DataFrame({"d": ["2025-01-01", "2024-01-01"]})
    out = utils.time_decay_weight(
        df,
        date_col="d",
        half_life_years=1.0,
        reference_date=pd.Timestamp("2025-01-01"),
    )
    assert len(out) == 2
    assert float(out.iloc[0]) == 1.0
    # Our implementation uses days/365.25, so a leap year won't be exactly 1.0 year.
    age_years = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 365.25
    expected = 0.5 ** (age_years / 1.0)
    assert float(out.iloc[1]) == pytest.approx(expected)


# Does the time_decay_weight function raise a ValueError when the dates are unparseable?
# must raise a ValueError when the dates are unparseable
def test_time_decay_weight_raises_value_error_on_unparseable_dates():
    df = pd.DataFrame({"d": ["not-a-date", "2025-01-01"]})
    with pytest.raises(ValueError, match="invalid d values"):
        utils.time_decay_weight(df, date_col="d")


def test_stable_schema_hash_order_insensitive_and_stable():
    h1 = utils.stable_schema_hash(["b", "a"])
    h2 = utils.stable_schema_hash(["a", "b"])
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64  # sha256 hex


def test_stable_schema_hash_empty_raises_value_error():
    with pytest.raises(ValueError):
        utils.stable_schema_hash([])


def test_uct_now_iso_returns_isoformat_with_timezone():
    s = utils.uct_now_iso()
    dt = pd.Timestamp(s)
    assert dt.tzinfo is not None


def test_normalize_dataframe_dtypes_converts_decimal_objects_to_float():
    df = pd.DataFrame({"x": [Decimal("1.5"), Decimal("2.0"), None]})
    out = utils.normalize_dataframe_dtypes(df)
    assert str(out["x"].dtype) == "Float64"
    assert float(out["x"].iloc[0]) == 1.5


def test_normalize_dataframe_dtypes_parses_numeric_strings_when_mostly_numeric():
    df = pd.DataFrame({"x": ["1", "2", "3", ""]})
    out = utils.normalize_dataframe_dtypes(df, numeric_parse_threshold=0.7)
    assert str(out["x"].dtype) == "Float64"
    assert float(out["x"].iloc[0]) == 1.0


def test_normalize_dataframe_dtypes_coerces_low_cardinality_strings_to_category():
    df = pd.DataFrame({"x": ["A", "A", "B", "B", "A"]})
    out = utils.normalize_dataframe_dtypes(df, numeric_parse_threshold=0.95, max_categories=10)
    assert str(out["x"].dtype) == "category"


def test_normalize_dataframe_dtypes_keeps_high_cardinality_strings_as_string_dtype():
    df = pd.DataFrame({"x": [f"v{i}" for i in range(100)]})
    out = utils.normalize_dataframe_dtypes(df, max_categories=10)
    # pandas "string" dtype prints as 'string' or 'string[python]' depending on version
    assert "string" in str(out["x"].dtype)


