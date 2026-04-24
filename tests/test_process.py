import pytest
import xarray as xr
from climate_risk.process import count_heat_days

def test_count_heat_days():
    data = xr.DataArray([15, 28, 32, 25, 31, 20, 29, 35, 19, 30])
    result = count_heat_days(data, threshold=30)
    assert result == 4
