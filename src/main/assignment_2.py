import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)

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
            first_multiplication = (x - x_points[i - j]) * matrix[i][j - 1]
            second_multiplication = (x - x_points[i]) * matrix[i - 1][j - 1]

            denominator = x_points[i] - x_points[i - j]

            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication) / denominator
            # print("Coefficient", coefficient)
            matrix[i][j] = coefficient

    # print(matrix)
    print(matrix[i][j])
    return None


# 2. Newton's Forward method
def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i + 1):
            numerator = matrix[i][j - 1] - matrix[i - 1][j - 1]
            denominator = x_points[i] - x_points[i - j]

            operation = numerator / denominator

            # cut it off to view it more simpler
            # matrix[i][j] = "{0:.7g}".format(operation)
            matrix[i][j] = operation

    # Print diagonal
    diagonal = []
    for i in range(1, size):
        diagonal.append(matrix[i][i])

    print()
    print(diagonal)

    return matrix


# 3. Approximation
def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]

    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]

        # we use the previous index for x_points....
        reoccuring_x_span *= value - x_points[index - 1]

        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span

        # add the reoccuring px result
        reoccuring_px_result += mult_operation

    # final result
    print()
    print(reoccuring_px_result)
    return reoccuring_px_result


# 4. Divided difference method, Hermite polynomial approximation
def hermite_interpolation(x_points, y_points, slopes):
    # matrix size changes because of "doubling" up info for hermite
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points * 2, num_of_points * 2))

    # populate x values (make sure to fill every TWO rows)
    for x in range(0, num_of_points * 2, 2):
        matrix[x][0] = x_points[int(x / 2)]
        matrix[x + 1][0] = x_points[int(x / 2)]
        # break

    # prepopulate y values (make sure to fill every TWO rows)
    for x in range(0, num_of_points * 2, 2):
        matrix[x][1] = y_points[int(x / 2)]
        matrix[x + 1][1] = y_points[int(x / 2)]

    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    for x in range(1, num_of_points * 2, 2):
        matrix[x][2] = slopes[int(x / 2)]

    filled_matrix = apply_div_dif(matrix)
    print()
    print(filled_matrix)


def apply_div_dif(matrix: np.array):
    size = len(matrix)

    for i in range(2, size):
        for j in range(2, i + 2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # get left cell entry
            left: float = matrix[i][j - 1]

            # get diagonal left entry
            diagonal_left: float = matrix[i - 1][j - 1]

            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i - j + 1][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation

    return matrix


# Q1 setup
x_points = [3.6, 3.8, 3.9]
y_points = [1.675, 1.436, 1.318]
approximating_value = 3.7
nevilles_method(x_points, y_points, approximating_value)

# Q2 setup
xi = [7.2, 7.4, 7.5, 7.6]
fxi = [23.5492, 25.3913, 26.8224, 27.4589]
divided_table = divided_difference_table(xi, fxi)

# Q3 find approximation
approximating_x = 7.3
final_approximation = get_approximate_result(divided_table, xi, approximating_x)

# Q4 setup
xi_points = [3.6, 3.8, 3.9]
yi_points = [1.675, 1.436, 1.318]
slopes = [-1.195, -1.188, -1.182]
# xi_points = [1.3, 1.6, 1.9]
# yi_points = [0.6200860, 0.4554022, 0.2818186]
# slopes = [-0.5220232, -0.5698959, -0.5811571]
hermite_interpolation(xi_points, yi_points, slopes)


# 5. Cubic spline interpolation
