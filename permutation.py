from itertools import permutations
from math import factorial

def invert_permutation(P):
    n = len(P)
    invP = [0] * n
    for i in range(n):
        invP[P[i]] = i
    return invP

def commutator(perm1, perm2):
    start = list(range(len(perm1)))

    start = [start[i] for i in perm1]
    start = [start[i] for i in perm2]
    start = [start[i] for i in invert_permutation(perm1)]
    start = [start[i] for i in invert_permutation(perm2)]

    return tuple(start)

for n in range(2, 6):

    print("for {} elements".format(n))

    sols = set()
    counter = 0
    for p1 in permutations(list(range(n))):
        for p2 in permutations(list(range(n))):
            sols.add(commutator(p1, p2))
            counter += 1

    print("{} possible permutations".format(factorial(n)))
    print("{} possible commutators".format(counter))
    print("resulting in {} permutations (commutators)".format(len(sols)))

    new_sols = set()
    for p1 in sols:
        for p2 in sols:
            new_sols.add(commutator(list(p1), list(p2)))
    
    print("resulting in {} permutations (commutators of commutators)".format(len(new_sols)))

    sols = new_sols
    new_sols = set()
    for p1 in sols:
        for p2 in sols:
            new_sols.add(commutator(list(p1), list(p2)))

    print("resulting in {} permutations (commutators of commutators of commutators)".format(len(new_sols)))

    sols = new_sols
    new_sols = set()
    for p1 in sols:
        for p2 in sols:
            new_sols.add(commutator(list(p1), list(p2)))

    print("resulting in {} permutations (commutators of commutators of commutators of commutators)".format(len(new_sols)))