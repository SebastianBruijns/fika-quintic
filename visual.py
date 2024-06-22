from aberth_method.function import Function
from aberth_method.aberthMethod import aberthMethod
import matplotlib.pyplot as plt
import numpy as np
import imageio
import math
import cmath

plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["FreeMono"]
title_size = 24
ticksize = 14
annotation_size = 14
trace_alpha = 0.3

a = Function({0: -1, 1: 1/2, 2:1})
aberthMethod(a)

coef_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628']
sol_colors = ['#984ea3', '#999999', '#e41a1c', '#dede00', 'k']

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

def plot_pol_and_roots(func, title, prev_sol, coef_traces=None, sol_traces=None, plot_start=False):
    plt.figure(figsize=(14, 6.5))
    plt.subplot(121)

    right_title = "=0"
    for i, (coef, color) in enumerate(zip(func.coef[:-1], coef_colors)):  # ignore last coefficient, it's 1, we ignore constant scaling
        plt.plot(coef.real, coef.imag, c=color, marker='o', ms=6)
        plt.annotate(r"$a_{}$".format(i), (coef.real + 0.15, coef.imag + 0.15), size=annotation_size)

        constant_sign = "+" if coef.real >= 0 else "-"
        mult = -1 if coef.real < 0 else 1
        x_string = "x^{}".format(i)
        if i == 1:
            x_string = "x"
        elif i == 0:
            x_string = ""
        right_title = "{}({:.1f}){}".format(constant_sign, (abs(coef.real) + mult * coef.imag * 1j), x_string) + right_title

        if coef_traces:
            coef_traces[i].append(coef)
            plt.plot([t.real for t in coef_traces[i]], [t.imag for t in coef_traces[i]], c=color, alpha=trace_alpha)

    if len(func.coef) < 4:
        plt.text(0.5, 1.06, ("x^2" + right_title).replace('j', 'i'),
            horizontalalignment='center',
            fontsize=title_size,
            transform = plt.gca().transAxes)

    plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)
    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_aspect('equal', 'box')

    if len(func.coef) < 4:
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
    else:
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

    plt.subplot(122)
    _, sols = aberthMethod(func)
    sols = sort_with_noise(prev_sol, sols)
    left_title = ""

    for i, (s, color) in enumerate(zip(sols, sol_colors)):

        plt.plot(s.real, s.imag, c=color, marker='*', ms=9)

        if sol_traces:
            sol_traces[i].append(s)
            if plot_start:
                plt.plot(sol_traces[i][0].real, sol_traces[i][0].imag, c=color, marker='o', ms=14, zorder=0)
                plt.plot(sol_traces[i][0].real, sol_traces[i][0].imag, c='white', marker='o', ms=11, zorder=0)
            plt.plot([t.real for t in sol_traces[i]], [t.imag for t in sol_traces[i]], c=color, alpha=trace_alpha)

        constant_sign = "+" if s.real < 0 else "-"  # sign flip, since we need to subtract the root
        left_title += "(x {} {:.1f})".format(constant_sign, (abs(s.real) + s.imag * 1j))
    left_title += "=0"

    if len(func.coef) < 4:
        # plt.title(left_title.replace('j', 'i'), size=title_size)
        plt.text(0.5, 1.06, left_title.replace('j', 'i'),
            horizontalalignment='center',
            fontsize=title_size,
            transform = plt.gca().transAxes)

    plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)
    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_aspect('equal', 'box')
        
    if len(func.coef) < 4:
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
    else:
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

    plt.tight_layout()
    plt.savefig("./images/" + str(title))
    plt.close()

    return sols, coef_traces, sol_traces

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

def interpolate_arc(p1, p2, bulge, t):
    """Interpolate along an arc between p1 and p2 with a bulge factor at t (0 <= t <= 1)"""
    if bulge == 0:
        # When bulge is zero, it degenerates to a straight line interpolation
        return interpolate(p1, p2, t)
    
    # Calculate the chord length and the sagitta (height of the arc)
    chord_length = abs(p2 - p1)
    sagitta = (bulge * chord_length) / 2

    # Calculate the midpoint of the chord
    midpoint = (p1 + p2) / 2

    # Calculate the angle between the chord and the x-axis
    angle = cmath.phase(p2 - p1)

    # Calculate the center of the circle forming the arc
    direction = cmath.exp(1j * (angle + math.pi / 2))  # Perpendicular direction to the chord
    arc_center = midpoint + sagitta * direction

    # Calculate the radius of the circle
    radius = abs(arc_center - p1)

    # Calculate the angle subtended by the arc
    theta = 2 * math.asin(chord_length / (2 * radius))

    # Calculate the angle for the interpolation point
    if bulge > 0:
        # Positive bulge means counterclockwise arc
        start_angle = cmath.phase(p1 - arc_center)
        interp_angle = start_angle + t * theta
    else:
        # Negative bulge means clockwise arc
        start_angle = cmath.phase(p2 - arc_center)
        interp_angle = start_angle - t * theta

    # Calculate the interpolated point on the arc
    interpolated_point = arc_center + radius * cmath.exp(1j * interp_angle)
    return interpolated_point

def arc_polygon(points, n, bulge):
    if n <= 0 or not points or len(points) < 2:
        return []
    
    result = [points[0]]  # Start with the first point
    total_points = len(points)
    
    for i, b in zip(range(total_points - 1), bulge):
        p1 = points[i]
        p2 = points[i + 1]
        
        for j in range(1, n // (total_points - 1)):
            t = j / (n // (total_points - 1))
            result.append(interpolate_arc(p1, p2, b, t))
    
    if len(result) < n:
        result.append(points[-1])  # Ensure the last point is included
    
    return result

def make_plots(points_to_traverse, title, traces=False, coefs=[-1, 1/2, 1], loop=None, plot_start=False):
    _, sols = aberthMethod(Function(dict(zip(range(len(coefs)), coefs))))
    sols = sorted(sols, key=lambda x: x.real)
    coef_traces = [[] for _ in range(len(coefs) - 1)]
    sol_traces = [[] for _ in range(len(coefs) - 1)]

    for i, points in enumerate(zip(*points_to_traverse)):
        mod_coefs = coefs.copy()
        mod_coefs = [c + p for c, p in zip(coefs, points)] + [1]
        if not traces:
            sols, coef_traces, sol_traces = plot_pol_and_roots(Function(dict(zip(range(len(coefs)), mod_coefs))),
                                                            title=title.format(i), prev_sol=sols, plot_start=plot_start)
        else:
            sols, coef_traces, sol_traces = plot_pol_and_roots(Function(dict(zip(range(len(coefs)), mod_coefs))),
                                                            title=title.format(i), prev_sol=sols, coef_traces=coef_traces,
                                                            sol_traces=sol_traces, plot_start=plot_start)

    images = []
    for i in range(len(points_to_traverse[0])):
        images.append(imageio.imread("./images/" + title.format(i) + '.png'))
    if loop is not None:
        imageio.mimsave("./gifs/" + 'gif_' + title[:-2] + '.gif', images, format='GIF', duration=0.065, loop=loop)
    else:
        imageio.mimsave("./gifs/" + 'gif_' + title[:-2] + '.gif', images, format='GIF', duration=0.065)

if False:
    triangle_points = polygon([0, -3+3j], 30)
    make_plots(points_to_traverse=[[0]*30, triangle_points], title="param_exp_1_{}")

    triangle_points = polygon([0, 0-2j], 30)
    make_plots(points_to_traverse=[triangle_points, [-3+3j]*30], title="param_exp_2_{}")

    triangle_points = polygon([-3+3j, 0+3j], 30)
    make_plots(points_to_traverse=[[0-2j]*30, triangle_points], title="param_exp_3_{}")

    make_plots(points_to_traverse=[polygon([0-2j, 0], 30), polygon([0+3j, 0], 30)], title="param_exp_4_{}")

# Example usage:
if False:
    triangle_points = polygon([0, -3+3j, +3j, 0], 60)
    make_plots(points_to_traverse=[[0]*60, triangle_points], title="loop_{}", traces=True)

    triangle_points = polygon([0, -3+3j, +3j, 0, -3+3j, +3j, 0], 120)
    make_plots(points_to_traverse=[[0]*120, triangle_points], title="double_loop_{}", traces=True, loop=0)

if False:
    p1 = [np.exp(2 * math.pi * 1j * x / 20) * 0.8 * min(1, (-abs(30 - x)+30) / 10) for x in range(60)]
    make_plots(points_to_traverse=[p1, [0]*60], title="continuous_change_{}", loop=0)

    p1 = [-np.exp(math.pi * 1j * x / 30) * 2 * (1 - abs(30 - x) / 30) for x in range(60)]
    p1 = p1 + [0] * 10 + p1 + [0] * 10
    make_plots(points_to_traverse=[p1, [0]*140], title="swap_{}", loop=0)

if False:
    p1 = [-np.exp(2 * math.pi * 1j * x / 60) * 2 for x in range(60)]
    make_plots(points_to_traverse=[p1, [0]*60], coefs=[0, 0, 1], title="continuous_change_2_{}", loop=0)


    p1 = [-np.exp(2 * math.pi * 1j * x / 60) * 2 for x in range(60)]
    p1 = p1 + [-2] * 10 + p1 + [-2] * 10
    make_plots(points_to_traverse=[p1, [0]*140], coefs=[0, 0, 1], title="swap_2_{}", loop=0)


# arc_polygon
a, b, c = 0.98+0.63j, -0.48+0.41j, 0.86-0.68j
# roots = [arc_polygon([a, b, b+0.0001], 100, 0.5), arc_polygon([b, a, c], 100, 0.5), arc_polygon([c, c+0.0001, a], 100, 0.5)]

# combined_roots = np.vstack(roots).T
# points_to_traverse = [np.poly(r) for r in combined_roots]
# points_to_traverse = list(np.array(points_to_traverse).T[1::][::-1])

# make_plots(points_to_traverse=points_to_traverse, coefs=[0, 0, 0, 1], title="cubic_{}", traces=True, plot_start=True)


roots = [arc_polygon([a, b, b+0.0001, a], 100, [0.5, 0.5, -0.5]), arc_polygon([b, a, c, a], 100, [0.5, 0.5, -0.5]), arc_polygon([c, c+0.0001, a, a+0.001], 100, [0.5, 0.5, -0.5])]

combined_roots = np.vstack(roots).T
points_to_traverse = [np.poly(r) for r in combined_roots]
points_to_traverse = list(np.array(points_to_traverse).T[1::][::-1])

make_plots(points_to_traverse=points_to_traverse, coefs=[0, 0, 0, 1], title="cubic_full_{}", traces=True, plot_start=True)
quit()
a, b, c, d, e = 1.55+1.45j, 0.03+0.91j, 1.21+0.02j, -1.13+0.37j, 0.21-0.95j
roots = [arc_polygon([a, b, b+0.0001], 100, 0.5), arc_polygon([b, a, c], 100, 0.5), arc_polygon([c, c+0.0001, a], 100, 0.5),
         np.ones(100) * d, np.ones(100) * e]

combined_roots = np.vstack(roots).T
points_to_traverse = [np.poly(r) for r in combined_roots]
points_to_traverse = list(np.array(points_to_traverse).T[1::][::-1])

make_plots(points_to_traverse=points_to_traverse, coefs=[0, 0, 0, 0, 0, 1], title="test_{}", traces=True, plot_start=True)


quit()
for i in range(40):
    plot_pol_and_roots(Function({0: -1, 1: 1/2 + np.exp(2 * 3.14159 * 1j * i / 40) * 5 * (1 - np.abs(20 - i) / 20), 2:1}), title=i)