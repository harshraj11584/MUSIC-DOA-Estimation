#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


def real_signal():
    # =========  READ Cal DATA from File========= %
    datafile = 'caldata32_1.csv';
    calData = np.genfromtxt(datafile, delimiter=',');
    CalData = calData[:,1]+1j*calData[:,2];    
    mCalData        =   np.tile(CalData,(N,1));
    # ========= READ INPUT DATA from File========= %
    datafile = 'calipeda1.csv';
    DataV = np.genfromtxt(datafile, delimiter=',');
    #applying calibration
    DataV32 = DataV*mCalData;
    DataV31 = np.concatenate((DataV32[:,0:16], DataV32[:,17:32]), axis=1);
    
    
    return np.transpose(DataV31);


# In[3]:


d = 2.0
c = 340.0
N = 1900
I = 5


# In[4]:


# X = pd.read_csv('caldata32_3.csv').to_numpy()
X = real_signal()
print("X=\n",X)

S = np.cov(X.T)
print("S=\n",S)


# In[ ]:


eigvals, eigvecs = np.linalg.eig(S)


# In[ ]:


#Sorting eigvals and eigvecs from largest to smallest

idx = eigvals.argsort()[::-1]   
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]
print("eigvals=",eigvals)
print("eigvecs=\n",eigvecs)


# In[ ]:


lb_min = np.min(eigvals)
lb_mean = np.mean(eigvals)
lb_std = (np.var(eigvals))**0.5

print(lb_min, lb_mean, lb_std)


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
ax.scatter(np.arange(1,len(eigvals)+1),eigvals)


# In[ ]:


I = 3
print("max eig vec : \n", eigvecs[0])
U = np.array(eigvecs).T
print("U=\n",U.shape)
Us, Un = U[:,:I], U[:,I:]
print("Us=\n",Us.shape)
print("Un=\n",Un.shape)


# In[ ]:


w = 2.0
N = len(eigvals)
def a(theta):
    a1 = np.exp(-1j*w*d*(np.sin(theta)/c) * np.arange(N) )
    return a1.reshape((N,1))

print(a(2))


# In[ ]:


def P_MU(theta):
    return complex(a(theta).conj().T @ Un @ Un.conj().T @ a(theta)).real
    
print(P_MU(2))


# In[ ]:


theta_vals = np.linspace(-200,200,1000)
P_MU_vals = np.array([P_MU(val) for val in theta_vals])
print(P_MU_vals)


# In[ ]:


plt.plot(theta_vals,P_MU_vals)


# In[ ]:




