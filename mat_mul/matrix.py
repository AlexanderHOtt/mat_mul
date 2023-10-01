from __future__ import annotations
from typing import List, Tuple, Union

class Matrix:
    def __init__(self, rows: int, cols: int, data: List[List[Union[int, float]]]):
        self.rows = rows
        self.cols = cols
        self.data = data

    @classmethod
    def from_list(cls, data: List[List[Union[int, float]]]) -> Matrix:
        rows = len(data)
        cols = len(data[0])
        return cls(rows, cols, data)

    def __str__(self) -> str:
        if not self.rows:
            return "[Empty]"
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
        row, col = key
        return self.data[row][col]

    def __setitem__(self, key: Tuple[int, int], value: Union[int, float]) -> None:
        row, col = key
        self.data[row][col] = value

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions are incompatible for multiplication")
        
        result_data: list[list[int|float]] = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result_data[i][j] += self.data[i][k] * other.data[k][j]

        return Matrix(self.rows, len(result_data[0]), result_data)

    def __mul__(self, other: Union[int, float, Matrix]) -> Matrix:
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result_data = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
        elif isinstance(other, Matrix):
            # Matrix multiplication
            return self @ other
        else:
            raise TypeError("Unsupported type for multiplication")

        return Matrix(self.rows, len(result_data[0]), result_data)

    def row_reduction(self) -> Matrix:
        result_data = [row[:] for row in self.data]  # Create a copy of the matrix data

        row, col = 0, 0
        while row < self.rows and col < self.cols:
            # Find the pivot row, which is the row with the largest absolute value in the current column
            pivot_row = max(range(row, self.rows), key=lambda i: abs(result_data[i][col]))

            # Swap the current row with the pivot row if necessary
            if pivot_row != row:
                result_data[row], result_data[pivot_row] = result_data[pivot_row], result_data[row]

            # Make the diagonal element 1 (scaling)
            pivot_element = result_data[row][col]
            if pivot_element != 0:
                scale_factor = 1.0 / pivot_element
                result_data[row] = [elem * scale_factor for elem in result_data[row]]

            # Eliminate non-zero elements below the pivot
            for i in range(row + 1, self.rows):
                factor = result_data[i][col]
                for j in range(col, self.cols):
                    result_data[i][j] -= factor * result_data[row][j]

            row += 1
            col += 1

        return Matrix(self.rows, self.cols, result_data)

    def reduced_row_echelon_form(self) -> Matrix:
        # Start with row reduction to bring the matrix to row echelon form
        row_echelon = self.row_reduction()

        result_data = [row[:] for row in row_echelon.data]

        for row in range(self.rows - 1, -1, -1):
            # Find the first non-zero element in the current row (the pivot element)
            pivot_col = next((col for col, value in enumerate(result_data[row]) if value != 0), None)
            
            if pivot_col is not None:
                # Make the pivot element 1
                pivot_element = result_data[row][pivot_col]
                result_data[row] = [elem / pivot_element for elem in result_data[row]]

                # Eliminate non-zero elements above the pivot
                for i in range(row - 1, -1, -1):
                    factor = result_data[i][pivot_col]
                    for j in range(pivot_col, self.cols):
                        result_data[i][j] -= factor * result_data[row][j]

        return Matrix(self.rows, self.cols, result_data)

