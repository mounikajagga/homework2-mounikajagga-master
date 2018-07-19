
# coding: utf-8

# In[64]:

import numpy as np
import math
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[65]:

x = np.array(([0.05, 0.1]))
y = np.array(([[0.01], 
               [0.99]]))


# In[66]:

alpha = 0.5


# In[67]:

th1 = np.random.random((2,3))
th2 = np.random.random((2,3))


# In[68]:

#setting a1
a1 = np.array([1.0,x[0],x[1]])


# In[69]:

indexerations = 200
costs=[]


# In[70]:

def sig(x):   
    return 1/(1+np.exp(-x))


# In[71]:

for i in range(indexerations):
    #forward propagation
    z2 = np.matmul(th1,a1.transpose())
    a2 = np.array([1.0,sig(z2[0]),sig(z2[1])])
    z3 = np.matmul(th2,a2.transpose())
    a3 = np.array([sig(z3[0]),sig(z3[1])])
    
    # Calculating cost
    cost= (1/2)*(math.pow(y[0]-a3[0],2) + math.pow(y[1]-a3[1],2))
    costs.append(cost)
    print(cost)
    
    #backward propogation
    der3_z1 = (a3[0]-y[0])*(a3[0]*(1-a3[0]))
    par_theta1_2 = der3_z1*a2
    der3_z2 = (a3[1]-y[1])*(a3[1]*(1-a3[1]))
    par_theta2_2 = der3_z2*a2
    par_theta_2 = np.array([par_theta1_2,par_theta2_2])
    th2 = th2 - (alpha * par_theta_2)
    
    J1_z1 = (a3[0]-y[0])*(a3[0]*(1-a3[0]))*th2[0][1]*(a2[0]*(1-a2[0]))
    J2_z1 = (a3[1]-y[1])*(a3[1]*(1-a3[1]))*th2[1][1]*(a2[0]*(1-a2[0]))
    J_z1 = J1_z1 + J2_z1
    par_theta1_1 = J_z1*a1
    J1_z2 = (a3[0]-y[0])*(a3[0]*(1-a3[0]))*th2[0][1]*(a2[1]*(1-a2[1]))
    J2_z2 = (a3[1]-y[1])*(a3[1]*(1-a3[1]))*th2[1][1]*(a2[1]*(1-a2[1]))
    J_z2 = J1_z2 + J2_z2
    par_theta2_1 = J_z2*a1
    par_theta_1 = np.array([par_theta1_1,par_theta2_1])
    th1 = th1 - (alpha * par_theta_1)

    #alpha = 2
    #th2 = th2 + alpha * (der2 * 2)
    #th1 = th1 + alpha * (djdt1 * x)


# In[62]:

len(costs)


# In[72]:

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Cost vs Number of iterations")
ax.plot(range(0, indexerations), costs)
ax.set_xlabel('Itertaions')
ax.set_ylabel('Cost')
fig.show()

