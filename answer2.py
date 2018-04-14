import numpy as np
from numpy import genfromtxt
csv = genfromtxt('P2_data/P2_train.csv', delimiter=',')
X = np.zeros((310,2))
Y = np.zeros(310)
for i in range(0,310):
	X[i] = csv[i][0:2]
	Y[i] = csv[i][2]
csv_test = genfromtxt('P2_data/P2_test.csv', delimiter=',')
Xt = np.zeros((90,2))
Yt = np.zeros(90)
for i in range(0,90):
	Xt[i] = csv_test[i][0:2]
	Yt[i] = csv_test[i][2]

no_of_1s = 0
no_of_0s = 0
for i in range(0,310):
	if(Y[i] == 1.0):
		no_of_1s += 1
	else:
		no_of_0s += 1

X0 = np.zeros((no_of_0s,2))
X1 = np.zeros((no_of_1s,2))
x0 = 0
x1 = 0
for i in range(0,310):
	if(Y[i] == 1.0):
		X1[x1] = X[i]
		x1 += 1
	else:
		X0[x0] = X[i]
		x0 += 1

mean0 = 0.0
for i in range(0,no_of_0s):
	mean0 += X0[i]
mean0 = mean0 / no_of_0s

mean1 = 0.0
for i in range(0,no_of_1s):
	mean1 += X1[i]
mean1 = mean1 / no_of_1s

var0 = 0.0
for i in range(0,no_of_0s):
	var0 += ((X0[i] - mean0)**2)
var0 = var0 / no_of_0s

var1 = 0.0
for i in range(0,no_of_1s):
	var1 += ((X1[i] - mean1)**2)
var1 = var1 / no_of_1s

cov_a_a = np.zeros((2,2))
cov_a_a[0][0] = (var1[0]+var0[0])/2.0
cov_a_a[1][1] = (var1[1]+var0[1])/2.0

inv_cov_a_a = np.linalg.inv(cov_a_a)
det_cov_a_a = np.linalg.det(cov_a_a)

# prediction
result0 = np.zeros(90)
result1 = np.zeros(90)
for i in range(0,90):
	u = Xt[i] - mean0
	beta = np.matmul(inv_cov_a_a, u)
	alpha = np.dot(u, beta)
	alpha = alpha * (-0.5)
	denominator = (6.28) * (det_cov_a_a**0.5)
	result0[i] = (2.71**alpha) / denominator

for i in range(0,90):
	u = Xt[i] - mean1
	beta = np.matmul(inv_cov_a_a, u)
	alpha = np.dot(u, beta)
	alpha = alpha * (-0.5)
	denominator = (6.28) * (det_cov_a_a**0.5)
	result1[i] = (2.71**alpha) / denominator

pred = np.zeros(90)
for i in range(0,90):
	if(result0[i] >= result1[i]):
		pred[i] = 0.0
	if(result1[i] >= result0[i]):
		pred[i] = 1.0

correct = 0.0
for i in range(0,90):
	if(pred[i] == Yt[i]):
		correct += 1.0

print str("Covarience matrix :\n")+str(cov_a_a)
print str("Accuracy : ")+str(float(correct)/90.0)

conf_mat = np.zeros((2,2))
for i in range(0,90):
	if((Yt[i] == 1) and (pred[i] == 1)):
		conf_mat[0][0] += 1.0
	if((Yt[i] == 1) and (pred[i] == 0)):
		conf_mat[0][1] += 1.0
	if((Yt[i] == 0) and (pred[i] == 1)):
		conf_mat[1][0] += 1.0
	if((Yt[i] == 0) and (pred[i] == 0)):
		conf_mat[1][1] += 1.0

print "Confusion matrix is:"
print conf_mat

cov_a_a[0][0] = (cov_a_a[0][0]+cov_a_a[1][1])/2.0
cov_a_a[1][1] = cov_a_a[0][0]

inv_cov_a_a = np.linalg.inv(cov_a_a)
det_cov_a_a = np.linalg.det(cov_a_a)

# prediction
result0 = np.zeros(90)
result1 = np.zeros(90)
for i in range(0,90):
	u = Xt[i] - mean0
	beta = np.matmul(inv_cov_a_a, u)
	alpha = np.dot(u, beta)
	alpha = alpha * (-0.5)
	denominator = (6.28) * (det_cov_a_a**0.5)
	result0[i] = (2.71**alpha) / denominator

for i in range(0,90):
	u = Xt[i] - mean1
	beta = np.matmul(inv_cov_a_a, u)
	alpha = np.dot(u, beta)
	alpha = alpha * (-0.5)
	denominator = (6.28) * (det_cov_a_a**0.5)
	result1[i] = (2.71**alpha) / denominator

pred = np.zeros(90)
for i in range(0,90):
	if(result0[i] >= result1[i]):
		pred[i] = 0.0
	if(result1[i] >= result0[i]):
		pred[i] = 1.0

correct = 0.0
for i in range(0,90):
	if(pred[i] == Yt[i]):
		correct += 1.0

print str("Covarience matrix :\n")+str(cov_a_a)
print str("Accuracy : ")+str(float(correct)/90.0)

conf_mat = np.zeros((2,2))
for i in range(0,90):
	if((Yt[i] == 1) and (pred[i] == 1)):
		conf_mat[0][0] += 1.0
	if((Yt[i] == 1) and (pred[i] == 0)):
		conf_mat[0][1] += 1.0
	if((Yt[i] == 0) and (pred[i] == 1)):
		conf_mat[1][0] += 1.0
	if((Yt[i] == 0) and (pred[i] == 0)):
		conf_mat[1][1] += 1.0

print "Confusion matrix is:"
print conf_mat

