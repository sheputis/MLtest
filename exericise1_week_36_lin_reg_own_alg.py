import numpy as np
import matplotlib.pyplot as plt
#from random import random,seed

#print(np.c_[np.array([1,2,3]),0,0, np.array([4,5,6])]) testing out np.c_ function
#print(np.c_[np.array([[1,2,3]]), np.array([[4,5,6]])])

x = 2*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)

xb = np.c_[np.ones((100,1)), x]
b  = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y) #beta vector

x_ =np.linspace(0,2,100)
y_ = b[0]+x_*b[1];

plt.plot(x,y,'ro')
plt.plot(x_,y_,'b-')
plt.show()
