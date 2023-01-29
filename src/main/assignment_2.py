import numpy as np
from decimal import Decimal, getcontext

EXPONENT_LENGTH = 11
MANTISSA_LENGTH = 52
MAX_ITERATIONS = 100

# Calculate decimal number using double precision (format to 5 decimal places)
def exercise_1(b_num: str):
    sign: int = 1 if b_num[0] == "0" else -1  # s = 0 the number is +
    exponent: float = 0
    mantissa: float = 0
    decimal_number: float = 0

    for i in range(1, EXPONENT_LENGTH + 1):
        exponent += int(b_num[i]) * (2 ** (11 - i))

    for i in range(EXPONENT_LENGTH + 1, len(b_num)):
        mantissa += int(b_num[i]) * (1 / 2) ** (i - 11)

    decimal_number = sign * (2 ** (exponent - 1023)) * (1 + mantissa)
    print(decimal_number)
    print()
    return decimal_number


def normalized_form(dec_num: float):
    n: int = 0
    # write number in normalized decimal form
    while dec_num >= 1:
        dec_num = dec_num / 10
        n += 1
    return (dec_num, n)


# Repeat Ex 1 using three-digit chopping arithmetic
def exercise_2(fraction: float, exponent: int, digits_to_chopping: int):
    chopped_value = int(fraction * 10**digits_to_chopping) / 10**digits_to_chopping
    return chopped_value * 10**exponent


# Repeat Ex 1 using three-digit rounding arithmetic
def exercise_3(fraction: float, exponent: int, digits_to_rounding: int):
    # add 5 to the (k+1) digit
    new_fraction = fraction + 5 / 10 ** (digits_to_rounding + 1)
    # then chop after the kth digit
    return exercise_2(new_fraction, exponent, digits_to_rounding)


# Absolute error with the exact value from question 1 and its 3 digit rounding
def absolute_error(precise: float, approximate: float):
    # sub_operation = precise - approximate
    sub_operation = approximate - precise
    print(abs(sub_operation))


# Relative error with the exact value from question 1 and its 3 digit rounding
def relative_error(precise: float, approximate: float):
    # print(getcontext())
    if precise != 0:
        rel_err = abs(Decimal(precise) - Decimal(approximate)) / abs(Decimal(precise))
        print(rel_err)
        print()


# Exercise 5 minimum number of terms needed to computer f(1) with error < 10^-4?
def check_for_alternating(series: str) -> bool:
    # check for (-1)^k term
    if "-1**k" in series:
        return True
    return False


def check_for_decreasing(series: str, x: int):
    k = 1
    previous_term = abs(eval(series))

    for k in range(2, 100):
        next_term = abs(eval(series))
        if previous_term <= next_term:
            return False
        previous_term = next_term
    return True


def compute_minimun_terms(error: float) -> int:
    min_number_terms = 0
    while (min_number_terms + 1) ** 3 < (1 / error):
        min_number_terms += 1
    print(min_number_terms)
    print()


# Exercise 6 number of iterations necessary to solve f(x) = x^3 + 4x^2 – 10 = 0 with accuracy 10^-4 using a = -4 and b = 7.

# Bisection method finds the first zero of any function to a certain error threshold
def bisection_method(left: float, right: float, tolerance: float, given_function: str):
    # Check f(a) and f(b) have different signs
    x = left
    intial_left = eval(given_function)
    x = right
    intial_right = eval(given_function)

    if intial_left * intial_right >= 0:
        print("Invalid inputs. Not on opposite sides of the function")
        return

    i = 0  # iteration counter
    while abs(right - left) > tolerance and i <= MAX_ITERATIONS:
        i += 1

        mid_point = (left + right) / 2
        x = mid_point
        evaluated_midpoint = eval(given_function)

        if evaluated_midpoint == 0.0:
            break

        # find function(left)
        x = left
        evaluated_left_point = eval(given_function)

        # this section basically checks if we have crossed the origin point (another way
        # to describe this is if f(midpoint) * f(left_point) changed signs)
        first_conditional: bool = evaluated_left_point < 0 and evaluated_midpoint > 0
        second_conditional: bool = evaluated_left_point > 0 and evaluated_midpoint < 0

        if first_conditional or second_conditional:
            right = mid_point
        else:
            left = mid_point

    print(i)
    print()


# Newton Raphson method finds a solution to f (x) = 0 given an initial approximation p0
def newton_raphson_method(initial_approximation: float, tolerance: float, given_function: str):

    i = 1
    function_derivative: str = "3*x**2 + 8*x"

    while i <= MAX_ITERATIONS:
        x = initial_approximation

        if eval(function_derivative) != 0:
            next_approximation = initial_approximation - eval(given_function) / eval(function_derivative)

            if abs(next_approximation - initial_approximation) < tolerance:
                print(i)
                print()
                return  # procedure was successfull

            i += 1
            initial_approximation = next_approximation
        else:
            print("Error derivative is zero")
            return

    print(f"The method failed after {MAX_ITERATIONS} number of iterations")


binary_number = "010000000111111010111001"

res1 = exercise_1(binary_number)

(fraction, exponent) = normalized_form(res1)

res2 = exercise_2(fraction, exponent, 3)
print(res2)
print()

res3 = exercise_3(fraction, exponent, 3)
print(res3)
print()

absolute_error(res1, res3)
relative_error(res1, res3)

series: str = "(-1**k) * (x**k) / (k**3)"
x: int = 1
error: float = 10 ** (-4)
check1: bool = check_for_alternating(series)
check2: bool = check_for_decreasing(series, x)
if check1 and check2:
    compute_minimun_terms(error)


left = -4
right = 7
error_tolerance: float = 10 ** (-4)
function_string = "x**3 + 4*(x**2) - 10"

bisection_method(left, right, error_tolerance, function_string)

initial_approximation: float = left
newton_raphson_method(initial_approximation, error_tolerance, function_string)
