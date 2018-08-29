import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,1)
y = 5*x*x+np.random.randn(100,1)

linreg = LinearRegression()
linreg.fit(x,y)
xnew = np.array([[0],[0.5],[1]])
ypredict = linreg.predict(xnew)
print(ypredict)
plt.plot(xnew,ypredict,'r-')
plt.plot(x,y,'ro')
plt.show()
#a goddamnt comment
