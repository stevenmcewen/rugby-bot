from datetime import date, timedelta

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