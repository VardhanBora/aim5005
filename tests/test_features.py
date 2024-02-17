import numpy as np
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the given vector
        """
        x = self._check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)


# Custom test for StandardScaler
def test_StandardScaler():
    # Test case 1: Testing standardization of a simple array
    scaler = StandardScaler()
    data = np.array([1, 2, 3, 4, 5])
    expected_mean = 3.0
    expected_std = np.std(data)
    expected_result = (data - expected_mean) / expected_std

    scaler.fit(data)
    assert np.allclose(scaler.mean, expected_mean)
    assert np.allclose(scaler.std, expected_std)

    transformed_data = scaler.transform(data)
    assert np.allclose(transformed_data, expected_result)

    # Test case 2: Testing standardization of a 2D array
    data = np.array([[1, 2], [3, 4], [5, 6]])
    expected_mean = np.mean(data, axis=0)
    expected_std = np.std(data, axis=0)
    expected_result = (data - expected_mean) / expected_std

    scaler.fit(data)
    assert np.allclose(scaler.mean, expected_mean)
    assert np.allclose(scaler.std, expected_std)

    transformed_data = scaler.transform(data)
    assert np.allclose(transformed_data, expected_result)

test_StandardScaler()
