import numpy as np
import matplotlib.pyplot as plt

x = 2*np.random.rand(100,1)
y = 4 + 3*x + np.random.rand(100,1)

plt.scatter(x, y)
plt.show()

x_b = np.c_[np.ones((100, 1)), x] #adding x0 in x

theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y) #normal equation
theta_best

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]

y_pred = x_new_b.dot(theta_best)
y_pred

plt.plot(x_new, y_pred, 'r-')
plt.plot(x, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

eta = 0.2
n_iteration = 1000
m = 100

theta = np.random.rand(2, 1) #random initialization

for iteration in range(n_iteration):
    gradient = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradient
    
theta
n_epochs = 50

t0, t1 = 5, 50 # hyperparameters

def learning_schedule(t):
    return t0/(t + t1)

theta = np.random.rand(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta*gradient
theta

