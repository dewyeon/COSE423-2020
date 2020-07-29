import numpy as np

# function: implementing gradient descent with backtracking line search
def min_gd(fun,x0,grad,args=()):
	x = x0
	alpha = 0.3
	beta = 0.8
	t = 1
	epsilon = 10e-5
	while np.linalg.norm(grad(x, *args), ord=2) > epsilon:
		# choose a direction
		delta_x = -grad(x, *args)
		# choose the step size: using backtracking line search
		while fun(x + t*delta_x, *args) >= fun(x, *args) + alpha*t*grad(x, *args).T@delta_x :
			t = beta*t
		# update x
		x = x + t*delta_x
	return x