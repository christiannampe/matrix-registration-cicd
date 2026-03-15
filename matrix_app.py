import re
import numpy as np

PHONE_PATTERN = r"^\d{3}-\d{3}-\d{4}$"
ZIP_PATTERN = r"^\d{5}-\d{4}$"


def validate_phone(phone):
    return re.fullmatch(PHONE_PATTERN, phone) is not None


def validate_zip(zip_code):
    return re.fullmatch(ZIP_PATTERN, zip_code) is not None


def create_matrix(rows):
    if len(rows) != 3:
        raise ValueError("Matrix must have exactly 3 rows.")

    matrix = []
    for row in rows:
        if len(row) != 3:
            raise ValueError("Each row must have exactly 3 values.")
        matrix.append([float(value) for value in row])

    return np.array(matrix, dtype=float)


def calculate(matrix_a, matrix_b, choice):
    if choice == "a":
        return matrix_a + matrix_b
    if choice == "b":
        return matrix_a - matrix_b
    if choice == "c":
        return np.matmul(matrix_a, matrix_b)
    if choice == "d":
        return matrix_a * matrix_b
    raise ValueError("Invalid operation choice.")


def get_row_means(matrix):
    return np.mean(matrix, axis=1)


def get_column_means(matrix):
    return np.mean(matrix, axis=0)


def get_transpose(matrix):
    return matrix.T


def main():
    print("========== Matrix Registration Application ==========")
    print("This app validates a phone number and ZIP+4, then performs matrix operations.")


if __name__ == "__main__":
    main()
