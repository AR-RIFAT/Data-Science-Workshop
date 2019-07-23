import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split

#observed predictors
x_train = np.array([1, 2, 3])
# or do this, which creates 3 x 1 vector so no need to reshape
#x_train = np.array([[1], [2], [3]])
print(x_train.shape)

x_train = x_train.reshape(len(x_train), 1)
#check dimensions
print(x_train.shape)

#observed responses
y_train = np.array([2, 2, 4])
# or do this, which creates 3 x 1 vector so no need to reshape
#y_train = np.array([[2], [2], [4]])
y_train = y_train.reshape(len(y_train),1)
print(y_train.shape)

#build matrix X by concatenating predictors and a column of ones
n = x_train.shape[0]
ones_col = np.ones((n, 1))
X = np.concatenate((ones_col, x_train), axis=1)
#check X and dimensions
print(X, X.shape)

#matrix X^T X
LHS = np.dot(np.transpose(X), X)

#matrix X^T Y
RHS = np.dot(np.transpose(X), y_train)

#solution beta to normal equations, since LHS is invertible by toy construction
betas = np.dot(np.linalg.inv(LHS), RHS)

#intercept beta0
beta0 = betas[0]

#slope beta1
beta1 = betas[1]

print(beta0, beta1)


def simple_linear_regression_fit(x_train, y_train):
    # your code here

    return betas


beta0 = simple_linear_regression_fit(x_train, y_train)[0]
beta1 = simple_linear_regression_fit(x_train, y_train)[1]

print("(beta0, beta1) = (%f, %f)" % (beta0, beta1))


#beta 1 > 0 which is reasonable given the data.  the best fit line should have a positive slope.
f = lambda x : beta0 + beta1*x
xfit = np.arange(0, 4, .01)
yfit = f(xfit)

plt.plot(x_train, y_train, 'ko', xfit, yfit)
plt.xlabel('x')
plt.ylabel('y')

#create the X matrix by appending a column of ones to x_train
X = sm.add_constant(x_train)
#this is the same matrix as in our scratch problem!
print(X)
#build the OLS model (ordinary least squares) from the training data
toyregr_sm = sm.OLS(y_train, X)
#save regression info (parameters, etc) in results_sm
results_sm = toyregr_sm.fit()
#pull the beta parameters out from results_sm
beta0_sm = results_sm.params[0]
beta1_sm = results_sm.params[1]

print("(beta0, beta1) = (%f, %f)" %(beta0_sm, beta1_sm))

#build the least squares model
toyregr_skl = linear_model.LinearRegression()
#save regression info (parameters, etc) in results_skl
results_skl = toyregr_skl.fit(x_train,y_train)
#pull the beta parameters out from results_skl
beta0_skl = results_skl.intercept_
beta1_skl = results_skl.coef_[0]

print("(beta0, beta1) = (%f, %f)" %(beta0_skl, beta1_skl))

#load mtcars
cars_data = pd.read_csv("data/mtcars.csv")
cars_data = cars_data.rename(columns={"Unnamed: 0":"name"})
cars_data.head()



#set random_state to get the same split every time
train_data, test_data = train_test_split(cars_data, test_size = 0.3, random_state = 6)



