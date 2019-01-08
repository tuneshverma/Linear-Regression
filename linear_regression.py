
# coding: utf-8

# In[96]:


import numpy as np
import matplotlib.pyplot as plt


# In[97]:


x = 2*np.random.rand(100,1)


# In[98]:


y = 4 + 3*x + np.random.rand(100,1)


# In[99]:


plt.scatter(x, y)


# In[100]:


plt.show()


# In[101]:


x_b = np.c_[np.ones((100, 1)), x] #adding x0 in x


# ### From Normal Equation

# In[102]:


theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y) #normal equation


# In[103]:


theta_best


# In[104]:


x_new = np.array([[0], [2]])


# In[105]:


x_new_b = np.c_[np.ones((2, 1)), x_new]


# In[106]:


y_pred = x_new_b.dot(theta_best)


# In[107]:


y_pred


# In[119]:


plt.plot(x_new, y_pred, 'r-')


# In[120]:


plt.plot(x, y, 'b.')


# In[121]:


plt.axis([0, 2, 0, 15])


# In[122]:


plt.show()


# ###  From Batch Gradient Decent

# In[123]:


eta = 0.2


# In[124]:


n_iteration = 1000


# In[125]:


m = 100


# In[127]:


theta = np.random.rand(2, 1) #random initialization


# In[128]:


for iteration in range(n_iteration):
    gradient = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradient


# In[129]:


theta


#  ### From Stochastic Gradient Decent

# In[130]:


n_epochs = 50


# In[131]:


t0, t1 = 5, 50 # hyperparameters


# In[132]:


def learning_schedule(t):
    return t0/(t + t1)


# In[133]:


theta = np.random.rand(2, 1)


# In[134]:


for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta*gradient


# In[135]:


theta

