import numpy as np


def f(x, n):
    ans = x
    for _ in range(n):
        ans = np.sqrt(ans + 1)

    return ans ** 2 - ans - 1

def g(x, n):
    ans = x
    for _ in range(n):
        ans = (ans + 1) ** (1 / 5)

    return ans ** 5 - ans - 1, ans


import matplotlib.pyplot as plt


a = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        a[i, j] = 1 if (((i-50)+(j-50)*1j)**5).real > 0 else 0

plt.imshow(a)
plt.show()