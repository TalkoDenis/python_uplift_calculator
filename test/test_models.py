import pytest

from src.models import LinearTrendForecaster
from src.exceptions import InsufficientDataError, NotFittedError

def test_calculate_coefficients_normal():
    x = [0, 1, 2, 3]
    y = [1.0, 3.0, 5.0, 7.0]

    alpha, beta = LinearTrendForecaster._calculate_coefficients(x, y)
    assert alpha == 1.0
    assert beta == 2.0


def test_calculate_coefficients_flat_line():
    x = [0, 1, 2]
    y = [5.0, 5.0, 5.0]

    alpha, beta = LinearTrendForecaster._calculate_coefficients(x, y)
    assert alpha == 5.0
    assert beta == 0.0


def test_calculate_coefficients_insuffient_data():
    x = [0]
    y = [10.0]

    with pytest.raises(InsufficientDataError):
        LinearTrendForecaster._calculate_coefficients(x, y)

def test_model_fit_predict_pipeline():
    model = LinearTrendForecaster(floor_value=None)
    x_train = [0, 1, 2]
    y_train = [0.0, 1.0, 2.0]
    model.fit(x_train, y_train)

    x_test = [3, 4]
    predictions = model.predict(x_test)

    assert predictions == [3.0, 4.0]

def test_model_predict_without_fit():
    model = LinearTrendForecaster()

    with pytest.raises(NotFittedError):
        model.predict([1, 2, 3])

def test_model_floor_values_clipping():
    model = LinearTrendForecaster()
    x_train = [0, 1, 2]
    y_train = [10.0, 8.0, 6.0]
    model.fit(x_train, y_train)

    predictions = model.predict([5, 6])

    assert predictions == [0.0, 0.0]
