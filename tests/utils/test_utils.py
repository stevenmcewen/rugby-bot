from datetime import date

import pytest

from functions.utils import utils


def test_to_date_range_single_day():
    start = date(2025, 12, 8)
    end = date(2025, 12, 8)

    result = utils.to_date_range(start, end)

    assert result == [start]


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


def test_to_date_range_start_after_end_raises_value_error():
    start = date(2025, 12, 10)
    end = date(2025, 12, 5)

    with pytest.raises(ValueError):
        utils.to_date_range(start, end)


