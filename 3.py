import numpy as np
from numpy import genfromtxt
csv = genfromtxt('Problem_3/Wage_dataset.csv', delimiter=',')
X_ = np.zeros(3000)

# wage vs age
for i in range(0,3000):
	X_[i] = csv[i][1]
r = np.zeros(3000)
for i in range(0,3000):
	r[i] = csv[i][10]

p = X_.argsort()
X_ = X_[p]
r = r[p]

D = np.zeros((3000,10))
for i in range(0,3000):
	for j in range(0,10):
		D[i][j] = X_[i]**j

L =  np.linalg.inv( np.matmul(D.T , D) )
R = np.matmul( D.T , r )
w = np.matmul(L, R)

wage_age = np.zeros(3000)
for i in range(0,3000):
	wage_age[i] = np.dot(w, D[i])

import matplotlib.pyplot as plt
plt.plot(X_,r,'ro',X_,wage_age)
plt.show()

# wage vs year

for i in range(0,3000):
	X_[i] = csv[i][0]
r = np.zeros(3000)
for i in range(0,3000):
	r[i] = csv[i][10]
p = X_.argsort()
X_ = X_[p]
r = r[p]

D = np.zeros((3000,10))
for i in range(0,3000):
	for j in range(0,10):
		D[i][j] = X_[i]**j

L =  np.linalg.inv( np.matmul(D.T , D) )
R = np.matmul( D.T , r )
w = np.matmul(L, R)

wage_year = np.zeros(3000)
for i in range(0,3000):
	wage_year[i] = np.dot(w, D[i])

plt.plot(X_,r,'ro',X_,wage_year)
plt.show()

# wage vs edu
for i in range(0,3000):
	X_[i] = csv[i][10]
r = np.zeros(3000)
for i in range(0,3000):
	r[i] = csv[i][4]



D = np.zeros((3000,10))
for i in range(0,3000):
	for j in range(0,10):
		D[i][j] = X_[i]**j

L =  np.linalg.inv( np.matmul(D.T , D) )
R = np.matmul( D.T , r )
w = np.matmul(L, R)

wage_edu = np.zeros(3000)
for i in range(0,3000):
	wage_edu[i] = np.dot(w, D[i])

# p = wage_edu.argsort()
# wage_edu = wage_edu[p]
# X_ = X_[p]

p = wage_edu.argsort()
wage_edu = wage_edu[p]
X_ = X_[p]
r = r[p]

plt.plot(r,X_,'ro',wage_edu,X_)
plt.show()