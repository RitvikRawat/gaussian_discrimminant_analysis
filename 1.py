import numpy as np
from numpy import genfromtxt
# from numpy.linalg import inv

X = genfromtxt('P1_data/P1_data_train.csv', delimiter=',')
Y = genfromtxt('P1_data/P1_labels_train.csv', delimiter=',')



#NOTATION: 
#			class 1 is the digit 5 
#			class 0 is the digit 6

# calculating phi
num_of_6s = 0
num_of_5s = 0
for i in range(0,777):
	if(Y[i] == 5.0):
		num_of_5s += 1
	else:
		num_of_6s += 1

phi =	num_of_5s/777.0

# calculating mean6
sum6 = np.zeros(64)
for i in range(0,777):
	if(Y[i] == 6.0):
		sum6 += X[i]
mean6 = sum6 / num_of_6s
# print mean6

# calculating mean5
sum5 = np.zeros(64)
for i in range(0,777):
	if(Y[i] == 5.0):
		sum5 += X[i]
mean5 = sum5 /	num_of_5s
# print mean5

# calculating covariance matrix class 5

X5 = np.zeros(	(num_of_5s,64))
X6 = np.zeros(	(num_of_6s,64))
i5 = 0
i6 = 0
for i in range(0,777):
	if(Y[i] == 5.0):
		X5[i5] = X[i] - mean5
		i5 += 1
	else:
		X6[i6] = X[i] - mean6
		i6 += 1

cov5 = np.zeros((64,64))
for i in range(0,64):
	for j in range(0,64):
		if(i == j):
			temp = 0.0
			for t in range(0,num_of_5s):
				temp += (X5[t][i])**2
			temp = temp / num_of_5s
			cov5[i][i] = temp
		else:
			temp = 0.0
			for t in range(0,num_of_5s):
				temp += np.dot( X5[t][i] , X5[t][j] )
			temp = temp / num_of_5s
			cov5[i][j] = temp

cov6 = np.zeros((64,64))
for i in range(0,64):
	for j in range(0,64):
		if(i == j):
			temp = 0.0
			for t in range(0,num_of_6s):
				temp += (X6[t][i])**2
			temp = temp / num_of_6s
			cov6[i][i] = temp
		else:
			temp = 0.0
			for t in range(0,num_of_6s):
				temp += np.dot( X6[t][i] , X6[t][j] )
			temp = temp / num_of_6s
			cov6[i][j] = temp

# print cov5
# print cov6

# testing phase
Xt = genfromtxt('P1_data/P1_data_test.csv', delimiter=',')
Yt = genfromtxt('P1_data/P1_labels_test.csv', delimiter=',')

result5 = np.zeros(333)
result6 = np.zeros(333)
cov5inv = np.linalg.inv(cov5)
cov6inv = np.linalg.inv(cov6)
det5 = np.linalg.det(cov5)
det6 = np.linalg.det(cov6)

for i in range(0,333):
	u = Xt[i] - mean5
	beta = np.matmul(cov5inv, u)
	alpha = np.dot(u, beta)
	alpha = alpha * (-0.5)
	denominator = (6.28**32) * (det5**0.5)
	result5[i] = (2.71**alpha) / denominator

for i in range(0,333):
	u = Xt[i] - mean6
	beta = np.matmul(cov6inv, u)
	alpha = np.dot(u, beta)
	alpha = alpha * (-0.5)
	denominator = (6.28**32) * (det6**0.5)
	result6[i] = (2.71**alpha) / denominator

# print result5
# print result6

pred = np.zeros(333)
for i in range(0,333):
	if(result5[i] >= result6[i]):
		pred[i] = 5.0
	if(result6[i] >= result5[i]):
		pred[i] = 6.0

correct = 0.0
for i in range(0,333):
	if(pred[i] == Yt[i]):
		correct += 1.0

print str("Accuracy is: ")+str(float(correct)/333.0)

conf_mat = np.zeros((2,2))
for i in range(0,333):
	if((Yt[i] == 5) and (pred[i] == 5)):
		conf_mat[0][0] += 1.0
	if((Yt[i] == 5) and (pred[i] == 6)):
		conf_mat[0][1] += 1.0
	if((Yt[i] == 6) and (pred[i] == 5)):
		conf_mat[1][0] += 1.0
	if((Yt[i] == 6) and (pred[i] == 6)):
		conf_mat[1][1] += 1.0

print "pie is : "+str(phi)
# print "mean of 5 is :"+str(mean5)
# print "mean of 6 is :"+str(mean6)
print "Confusion matrix is:"
print conf_mat

# print 1-(float(c5)/float(num_of_5s))
# print 1-(float(c6)/float(num_of_6s))