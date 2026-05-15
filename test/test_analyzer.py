import pytest
import datetime
from src.analyzer import UpliftAnalyzer
from src.exceptions import InsufficientDataError, DataValidationError

def test_full_pipeline_positive_uplift():
    """Интеграционный тест: идеальный линейный тренд с явным приростом."""
    raw_data = [
        (datetime.date(2026, 1, 1), 10.0),
        (datetime.date(2026, 1, 2), 20.0),
        (datetime.date(2026, 1, 3), 30.0),
        (datetime.date(2026, 1, 4), 40.0),
        (datetime.date(2026, 1, 5), 50.0),
        (datetime.date(2026, 1, 6), 60.0),
    ]

    raw_data.extend([
        (datetime.date(2026, 1, 7), 90.0),
        (datetime.date(2026, 1, 8), 100.0),
        (datetime.date(2026, 1, 9), 110.0),
        (datetime.date(2026, 1, 10), 120.0),
    ])

    analyzer = UpliftAnalyzer(
        forecast_model='linear_trend',
        smoothing_strategy=None
    )

    intervention_date = datetime.date(2026, 1, 7)
    result = analyzer.analyze(raw_data, intervention_date)

    assert result.total_uplift == 80.0
    assert round(result.relative_uplift_percentage, 2) == 23.53
    assert result.train_mae == 0.0

def test_analyzer_insufficient_data():
    """Проверка валидации длины данных внутри анализатора."""
    raw_data = [
        (datetime.date(2026, 1, 1), 10.0),
        (datetime.date(2026, 1, 2), 20.0),
    ]
    analyzer = UpliftAnalyzer()

    # Текст должен совпадать с тем, что в src/analyzer.py
    with pytest.raises(InsufficientDataError, match='Do not have enough data'):
        analyzer.analyze(raw_data, intervention_date=datetime.date(2026, 1, 2))

def test_full_pipeline_with_smoothing():
    """Проверяем работу пайплайна с включенным сглаживанием (moving_average)."""
    raw_data = [
        (datetime.date(2026, 1, 1), 10.0),
        (datetime.date(2026, 1, 2), 30.0),
        (datetime.date(2026, 1, 3), 20.0),
        (datetime.date(2026, 1, 4), 40.0),
    ]

    raw_data.extend([
        (datetime.date(2026, 1, 5), 50.0),
        (datetime.date(2026, 1, 6), 60.0),
    ])

    analyzer = UpliftAnalyzer(
        forecast_model='linear_trend',
        smoothing_strategy='moving_average',
        window_size=2
    )
    
    result = analyzer.analyze(raw_data, intervention_date=datetime.date(2026, 1, 5))
    
    assert isinstance(result.total_uplift, float)
    assert len(result.forecast_data) == 2
