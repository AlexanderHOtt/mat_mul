import numpy as np
from typing import List, Tuple, Union

class NPMatrix:
    def __init__(self, data: Union[List[List[Union[int, float]]], np.ndarray]):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=float)

    @classmethod
    def from_list(cls, data: List[List[Union[int, float]]]) -> "NPMatrix":
        return cls(data)

    # def __str__(self) -> str:
    #     return str(self.data)

    def __str__(self) -> str:
        num_rows = len(self.data)
        num_cols = len(self.data[0])

        max_width = max(len(str(self.data[i][j])) for i in range(num_rows) for j in range(num_cols))

        ret = ""
        for i in range(num_rows):
            for j in range(num_cols):
                # Right-align and pad each element to the maximum width
                ret += f"{self.data[i][j]:>{max_width}} "
            ret += "\n"

        return ret

    def __getitem__(self, key: Tuple[int, int]) -> Union[int, float]:
        return self.data[key]

    def __setitem__(self, key: Tuple[int, int], value: Union[int, float]) -> None:
        self.data[key] = value

    def __matmul__(self, other: "NPMatrix") -> "NPMatrix":
        # Matrix multiplication using NumPy's dot product
        return NPMatrix(np.dot(self.data, other.data))

    def __mul__(self, other: Union[int, float, "NPMatrix"]) -> "NPMatrix":
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result_data = self.data * other
        elif isinstance(other, NPMatrix):
            return self @ other
        else:
            raise TypeError("Unsupported type for multiplication")

        return NPMatrix(result_data)

    def row_reduction(self) -> "NPMatrix":
            # Implement Gaussian elimination for row reduction using NumPy

            matrix = self.data.copy()
            rows, cols = matrix.shape
            pivot_row, pivot_col = 0, 0

            while pivot_row < rows and pivot_col < cols:
                # Find the pivot element
                pivot_idx = np.argmax(np.abs(matrix[pivot_row:, pivot_col])) + pivot_row

                if matrix[pivot_idx, pivot_col] != 0:
                    # Swap rows if the pivot element is not zero
                    matrix[[pivot_row, pivot_idx]] = matrix[[pivot_idx, pivot_row]]

                    # Scale the pivot row to make the pivot element 1
                    matrix[pivot_row] /= matrix[pivot_row, pivot_col]

                    # Eliminate non-zero elements below the pivot
                    for i in range(pivot_row + 1, rows):
                        factor = matrix[i, pivot_col]
                        matrix[i] -= factor * matrix[pivot_row]

                    pivot_row += 1

                pivot_col += 1

            return NPMatrix(matrix)

    def reduced_row_echelon_form(self) -> "NPMatrix":
            # Implement reduced row echelon form (RREF) using NumPy

            matrix = self.data.copy()
            rows, cols = matrix.shape
            pivot_row, pivot_col = 0, 0

            while pivot_row < rows and pivot_col < cols:
                # Find the pivot element
                pivot_idx = np.argmax(np.abs(matrix[pivot_row:, pivot_col])) + pivot_row

                if matrix[pivot_idx, pivot_col] != 0:
                    # Swap rows if the pivot element is not zero
                    matrix[[pivot_row, pivot_idx]] = matrix[[pivot_idx, pivot_row]]

                    # Scale the pivot row to make the pivot element 1
                    matrix[pivot_row] /= matrix[pivot_row, pivot_col]

                    # Eliminate non-zero elements above and below the pivot
                    for i in range(rows):
                        if i != pivot_row:
                            factor = matrix[i, pivot_col]
                            matrix[i] -= factor * matrix[pivot_row]

                    pivot_row += 1

                pivot_col += 1

            return NPMatrix(matrix)

