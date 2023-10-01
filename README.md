# mat_mul

Matrix utilities for python. Implemented using both python lists and numpy arrays.

## Usage

## Example

```bash
git clone https://github.com/AlexanderHOtt/mat_mul.git
cd mat_mul
python -m mat_mul
```

## Example Code

<https://github.com/AlexanderHOtt/mat_mul/blob/main/mat_mul/__main__.py>

```python
from mat_mul import Matrix, NPMatrix


print("="*80)
print("Regular")
print("="*80)

matrix1 = Matrix.from_list([[1, 2, 3], [4, 5, 6]])
matrix2 = Matrix.from_list([[7, 8], [9, 10], [11, 12]])
result = matrix1 @ matrix2
print(result)

matrix = Matrix.from_list([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
reduced_matrix = matrix.row_reduction()
print(reduced_matrix)


print("="*80)
print("Numpy matricies")
print("="*80)

matrix1 = NPMatrix.from_list([[1, 2, 3], [4, 5, 6]])
matrix2 = NPMatrix.from_list([[7, 8], [9, 10], [11, 12]])
result = matrix1 @ matrix2
print(result)

matrix = NPMatrix.from_list([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
row_echelon_matrix = matrix.row_reduction()
print(row_echelon_matrix)

matrix = NPMatrix.from_list([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
rref_matrix = matrix.reduced_row_echelon_form()
print(rref_matrix)
```
