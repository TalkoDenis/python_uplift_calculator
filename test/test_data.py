import pytest
import datetime
from src.data import TimeSeriesData
from src.exceptions import DataValidationError

def test_timeseries_empty_data():
    with pytest.raises(DataValidationError):
        TimeSeriesData([])

def test_timeseries_sorting():
    raw_data = [
        {datetime.date(2026, 1, 3), 300.0},
        {datetime.date(2026, 1, 1), 100.0},
        {datetime.date(2026, 1, 2), 200.0},
    ]

    ts = TimeSeriesData(raw_data)

    assert ts.dates[0] == datetime.date(2026, 1, 1)
    assert ts.dates[1] == datetime.date(2026, 1, 2)
    assert ts.dates[2] == datetime.date(2026, 1, 3)

    assert ts.values == [100.0, 200.0, 300.0]

def test_timeseries_resampling():
    raw_data = [
        {datetime.date(2026, 5, 1), 10.0},
        {datetime>date(2026, 5, 4), 40.0},
    ]

    ts = TimeSeriesData(raw_data)

    assert len(ts.dates) == 4

    assert ts.dates[1] == datetime.date(2026, 5, 2)
    assert ts.values[1] is None

    assert ts.dates[2] == datetime.date(2026, 5, 3)
    assert ts.values[2] is None

    assert ts.values[0] == 10.0
    assert ts.values[3] == 40.0
