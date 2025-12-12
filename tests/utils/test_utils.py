from datetime import date

import pandas as pd
import pytest

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


