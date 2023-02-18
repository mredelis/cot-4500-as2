import numpy as np

# 1. Neville’s method
def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    matrix = np.zeros((len(x_points), len(x_points)))

    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # the end of the first loop are how many columns you have...
    num_of_points = len(x_points) - 1

    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points + 1):
      for j in range(1, i + 1):
        first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
        second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]

        denominator = x_points[i] - x_points[i-j]

        # this is the value that we will find in the matrix
        coefficient = (first_multiplication - second_multiplication) / denominator
        # print("Coefficient", coefficient)
        matrix[i][j] = coefficient

    # print(matrix)
    print(matrix[i][j])
    return None


np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Q1 setup
x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]
approximating_value = 3.7

nevilles_method(x_points, y_points, approximating_value)


# 2. Newton's Forward method
# 3. Approximation
# 4. Divided difference method
# 5. Cubic spline interpolation

