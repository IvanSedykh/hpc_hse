import numpy as np

tol = 1e-5
true_x_path = 'true_x.txt'
my_x_path = 'output.txt'

true_x = np.loadtxt(true_x_path, delimiter=' ')
my_x = np.loadtxt(my_x_path, delimiter=' ')

diff = np.linalg.norm(true_x - my_x)

print(f"diff norm= {diff}")
if diff < tol:
    print('OK')
    print(f'diff less than tol={tol}')
else:
    print('NOT OK')
    print(f'diff more than tol={tol}')
