import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
x = 2*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)

xb = np.c_[np.ones((100,1)), x]
b  = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y) #beta vector

x_ =np.linspace(0,2,100)
y_ = b[0]+x_*b[1];

print(mse(y,y_)) #y true y_ predict
print(r2s(y,y_))
