import numpy as np
import pytest

from app.matrix_app import (
    validate_phone,
    validate_zip,
    create_matrix,
    calculate,
    get_row_means,
    get_column_means,
    get_transpose
)


def test_validate_phone_valid():
    assert validate_phone("123-456-7890") is True


def test_validate_phone_invalid():
    assert validate_phone("1234567890") is False


def test_validate_zip_valid():
    assert validate_zip("21042-1234") is True


def test_validate_zip_invalid():
    assert validate_zip("210421234") is False


def test_create_matrix():
    rows = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    matrix = create_matrix(rows)
    assert matrix.shape == (3, 3)


def test_addition():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)
    result = calculate(a, b, "a")
    expected = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]], dtype=float)
    assert np.array_equal(result, expected)


def test_subtraction():
    a = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    result = calculate(a, b, "b")
    expected = np.array([[8, 6, 4], [2, 0, -2], [-4, -6, -8]], dtype=float)
    assert np.array_equal(result, expected)


def test_matrix_multiplication():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    result = calculate(a, b, "c")
    assert np.array_equal(result, a)


def test_elementwise_multiplication():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    b = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=float)
    result = calculate(a, b, "d")
    expected = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=float)
    assert np.array_equal(result, expected)


def test_invalid_operation():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    with pytest.raises(ValueError):
        calculate(a, b, "x")


def test_transpose():
    matrix = np.array([[1, 2], [3, 4]])
    result = get_transpose(matrix)
    expected = np.array([[1, 3], [2, 4]])
    assert np.array_equal(result, expected)


def test_row_means():
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    result = get_row_means(matrix)
    expected = np.array([2.0, 5.0])
    assert np.array_equal(result, expected)


def test_column_means():
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    result = get_column_means(matrix)
    expected = np.array([2.5, 3.5, 4.5])
    assert np.array_equal(result, expected)
