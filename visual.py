from aberth_method.function import Function
from aberth_method.aberthMethod import aberthMethod
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["FreeMono"]

a = Function({0: -1, 1: 1/2, 2:1})
aberthMethod(a)

colors = ['r', 'g', 'b']

def sort_with_noise(reference_list, noisy_list):
    sorted_noisy_list = []
    used_indices = set()

    for ref in reference_list:
        # Find the closest element in the noisy list that hasn't been used yet
        closest_index = None
        closest_distance = float('inf')
        
        for i, noisy in enumerate(noisy_list):
            if i in used_indices:
                continue

            distance = abs(ref - noisy)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        if closest_index is not None:
            sorted_noisy_list.append(noisy_list[closest_index])
            used_indices.add(closest_index)

    return sorted_noisy_list

def plot_pol_and_roots(func, title, prev_sol):
    plt.figure(figsize=(12, 8))
    plt.subplot(121)

    for coef, color in zip(func.coef[:-1], colors):  # ignore last coefficient, it's 1, we ignore constant scaling
        plt.plot(coef.real, coef.imag, c=color, marker='o')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    constant_sign = "+" if func.coef[-2].real >= 0 else "-"
    plt.title("x^2 + ({:.1f})x {} {:.1f}=0".format(func.coef[0], constant_sign, (abs(func.coef[1].real) + func.coef[1].imag * 1j)).replace('j', 'i'))

    plt.subplot(122)
    _, sols = aberthMethod(func)
    sols = sort_with_noise(prev_sol, sols)
    left_title = ""

    for s, color in zip(sols, colors):
        plt.plot(s.real, s.imag, c=color, marker='o')

        constant_sign = "+" if s.real >= 0 else "-"
        left_title += "(x {} {:.1f})".format(constant_sign, (abs(s.real) + s.imag * 1j))
    left_title += "=0"

    plt.title(left_title.replace('j', 'i'))
        
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig(str(title))
    plt.close()

    return sols

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

_, sols = aberthMethod(Function({0: -1, 1: 1/2, 2:1}))
sols = sorted(sols, key=lambda x: x.imag)

for i in range(60):
    sols = plot_pol_and_roots(Function({0: -1, 1: 1/2 + triangle_points[i], 2:1}), title=i, prev_sol=sols)

quit()
for i in range(40):
    plot_pol_and_roots(Function({0: -1, 1: 1/2 + np.exp(2 * 3.14159 * 1j * i / 40) * 5 * (1 - np.abs(20 - i) / 20), 2:1}), title=i)