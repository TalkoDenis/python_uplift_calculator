import pytest
from src.preprocessor import DataPreprocessor
from src.exceptions import DataValidationError

def test_imput_validation_empty_list():
    assert DataPreprocessor.impute_missing_values([]) == []

def test_impute_validation_all_notes():
    with pytest.raises(DataValidationError, match='All values are None'):
        DataPreprocessor.impute_missing_values([None, None, None])

def test_impute_validation_edges_none():
    with pytest.raises(DataValidationError, match='The first and the last'):
        DataPreprocessor.impute_missing_values([None, 10.0, 20.0])

    with pytest.raises(DataValidationError, match='The first and the last'):
        DataPreprocessor.impute_missing_values([10.0, 20.0, None])

def test_impute_unknown_strategy():
    with pytest.raises(ValueError, match='The wrong strategy'):
        DataPreprocessor.impute_missing_values([10.0, None, 20.0], strategy='magic_fill')

def test_impute_zero():
    data = [10.0, None, 20.0, None, 30.0]
    result = DataPreprocessor.impute_missing_values(data, strategy='zero')
    assert result == [10.0, 0.0, 20.0, 0.0, 30.0]

def test_impute_forward_fill():
    data = [10.0, None, None, 40.0]
    result = DataPreprocessor.impute_missing_values(data, strategy='forward_fill')
    assert result == [10.0, 10.0, 10.0, 40.0]

def test_impute_linear():
    data = [10.0, None, None, 40.0]
    result = DataPreprocessor.impute_missing_values(data, strategy='linear')
    assert result == [10.0, 20.0, 30.0, 40.0]

def test_impute_mean():
    data = [10.0, None, 30.0]
    result = DataPreprocessor.impute_missing_values(data, strategy='mean')
    assert result == [10.0, 20.0, 30.0]

def test_impute_median():
    data = [10.0, 100.0, None, 200.0]
    result = DataPreprocessor.impute_missing_values(data, strategy='median')
    assert result == [10.0, 100.0, 100.0, 200.0]
