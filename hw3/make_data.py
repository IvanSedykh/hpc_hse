import numpy as np

np.random.seed(11)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


N = 1_000
A = np.random.random(size=(N, N)) * 10
A = A @ A.T
assert is_pos_def(A)
assert np.allclose(A, A.T)

b = np.random.random(N) * 10

np.savetxt('input.txt', np.vstack([A, b]), delimiter=' ', fmt='%.16f')

x = np.linalg.solve(A, b)
np.savetxt('true_x.txt', x, delimiter=' ', fmt='%.12f')
