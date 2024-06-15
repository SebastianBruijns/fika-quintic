from aberth_method.function import Function
from aberth_method.aberthMethod import aberthMethod
import matplotlib.pyplot as plt
import numpy as np


a = Function({0: -1, 1: 1/2, 2:1})
aberthMethod(a)

def plot_pol_and_roots(func, title):
    plt.subplot(121)

    plt.plot([func.coef[0].real], [func.coef[0].imag], c='r', marker='o')
    plt.plot([func.coef[1].real], [func.coef[1].imag], c='g', marker='o')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.subplot(122)
    _, sols = aberthMethod(func)

    plt.plot([sols[0].real], [sols[0].imag], c='r', marker='o')
    plt.plot([sols[1].real], [sols[1].imag], c='g', marker='o')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.savefig(str(title))
    plt.close()

def interpolate(p1, p2, t):
    """ Interpolates between points p1 and p2 with a fraction t (0 <= t <= 1) """
    return p1 + (p2 - p1) * t

def polygon(points, n):
    if n <= 0 or not points or len(points) < 2:
        return []
    
    # Calculate edge lengths and the total perimeter
    edge_lengths = []
    total_length = 0
    for i in range(len(points) - 1):
        length = abs(points[i+1] - points[i])
        edge_lengths.append(length)
        total_length += length

    # Distance between each interpolated point along the perimeter
    segment_length = total_length / (n - 1)

    result = [points[0]]  # Start with the first point
    current_length = 0
    current_edge = 0
    next_point_idx = 1

    while len(result) < n:
        if next_point_idx >= len(points):
            break
        
        next_length = edge_lengths[current_edge]
        if current_length + next_length >= segment_length:
            remaining_length = segment_length - current_length
            t = remaining_length / next_length
            new_point = interpolate(points[current_edge], points[next_point_idx], t)
            result.append(new_point)
            points[current_edge] = new_point
            edge_lengths[current_edge] -= remaining_length
            current_length = 0
        else:
            current_length += next_length
            current_edge += 1
            next_point_idx += 1


    # Ensure the last point is included
    if len(result) < n:
        result.append(points[-1])

    return result

# Example usage:
points = [0 + 0j, 4 + 0j, 4 + 3j, 0 + 3j]
n = 10
triangle_points = polygon([0, -3+3j, +3j, 0], 60)

for i in range(60):
    plot_pol_and_roots(Function({0: -1, 1: 1/2 + triangle_points[i], 2:1}), title=i)

quit()
for i in range(40):
    plot_pol_and_roots(Function({0: -1, 1: 1/2 + np.exp(2 * 3.14159 * 1j * i / 40) * 5 * (1 - np.abs(20 - i) / 20), 2:1}), title=i)