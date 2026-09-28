# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(2)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)
def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):    
    return np.array_equal(h(w, X), y) 

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi: # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 
                w.append(w_new)

                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)
d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)

print("Final weight vector:")
print(w[-1])
print("Number of misclassified points recorded:", len(m))

# Plot the data and the final decision boundary
plt.figure(figsize=(6, 6))
plt.scatter(X[1, :N], X[2, :N], color='red', label='Class +1')
plt.scatter(X[1, N:], X[2, N:], color='blue', label='Class -1')

w_final = w[-1].ravel()
# Decision boundary: w0 + w1*x + w2*y = 0 -> y = -(w0 + w1*x) / w2
x_vals = np.linspace(0, 6, 100)
y_vals = -(w_final[0] + w_final[1] * x_vals) / w_final[2]
plt.plot(x_vals, y_vals, color='green', linewidth=2, label='Decision boundary')

plt.xlim(0, 6)
plt.ylim(0, 4)
plt.legend()
plt.title('Perceptron classification result')
plt.show()
