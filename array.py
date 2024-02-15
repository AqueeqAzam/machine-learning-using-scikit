
# Use numpy to create a one-dimensional array
import numpy as np

# create a vector as a row
"""vector_row = np.array([1, 2, 3])

# create a vector as a column
vector_col = np.array([[1],[2],[3]] )
print(vector_col)

# create matrix
matrix = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])
print(matrix)

# creating a sparse matrix
from scipy import sparse
matrix = np.array([[0, 0],
                  [0, 1],
                  [3, 0]])
print(matrix)

# select the elements
vector = np.array([2, 0, 9, 4, 5])
print(vector[2])
print(vector[:])    # selecting all elements
print(vector[:3])   # selecting everything up to and including the third element
print(vector[3:])   # selecting everything after the third element
print(vector[-1])   # selcting last elements

# describing matrix shape, size and dimentions
matrix = np.array([[0, 0, 7],
                  [0, 1, 5],
                  [3, 0, 2]])
print(matrix.shape)
print(matrix.size)
print(matrix.ndim)

# applying operation on elements
matrix = np.array([[23, 12, 43],
                   [12, 34, 56],
                   [76, 54, 43]])
print(matrix)

# create a function that adds 100 to somethings
add_100 = lambda i: i + 100

# create vectoized function
vectorized_add_100 = np.vectorize(add_100)

# apply function to all elements of matrix
vectorized_add_100(matrix)

# add 100 to all elements
print(matrix+100)

# find max and min value
matrix = np.array([[23, 12, 43],
                   [12, 34, 56],
                   [76, 54, 43]])
ma = np.max(matrix)
print('Maximum value:', ma)
mi = np.min(matrix)
print('Minimum value:', mi)

# find max element of column
ma_c = np.max(matrix, axis=0)
print(ma_c)

# find max element of row    
mi_c = np.max(matrix, axis=1)
print(mi_c)

# calculate the average, variance and standard deviation
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# return mean
mean = np.mean(matrix)
print(mean)

variance = np.var(matrix)
print(variance)

std_dev = np.std(matrix)
print(std_dev)

# reshape array
matrix = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
m_r = matrix.reshape(2, 6)
print(m_r)

# transpose a matrix
matrix = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])

# transpose matrix
transpose = matrix.T
print(transpose)

matrix = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])

flatten = matrix.flatten()
print(flatten)

# finding the rank of matrix
matrix = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])

# rank of matrix
# The rank of a matrix is the dimensions of the vector space spanned by its columns or rows.
rank = np.linalg.matrix_rank(matrix)
print(rank)

# calculating determinant, diagonal, trace
matrix = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])

det = np.linalg.det(matrix)
print(det)
print(matrix.diagonal())
print(matrix.trace())

# calculating dot product
# create two vector
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# calculate dot product
d_p = np.dot(vector_a, vector_b)
print(d_p)

# adding and substracting matrix
matrix_a = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])

matrix_b = np.array([[1, 2, 3], 
                  [4, 5, 6],
                  [7, 8, 9]])

mat_add = np.add(matrix_a, matrix_b)
mat_sub = np.sub(matrix_a, matrix_b)
print(mat_add, mat_sub)"""




