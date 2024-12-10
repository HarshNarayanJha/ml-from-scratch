# Implementing ML Algorithms from Scratch

This repo contains the implementation for a few basic ML algorithms from scratch in python

### 1. K-Nearest-Neighbours (KNN)

Given a data point

- Calculate its distance from all other data points (slow)
- Get the closest K points (hyperparameter, user-determined)
- _Regression_: Get the average of their values
- _Classification_: Get the label with majority vvote

Find the implementation in the file `knn.py`

### 2. Linear Regression

Understand the pattern/slope of the given dataset
Need to find a linear line (pun intended) that fits the data as closely as possible

$$ \hat{y} = wx + b $$

We also find the mean square error (MSE) for our regression.

$$ MSE = J(w, b) = \frac{1}{N} \sum_{i=1}^{n}{(y_i - (wx_i + b)^2)} $$

To find the best fitting line, we find the values for $w$ and $b$ that would give us the minimum MSE.

And to minimze something, we calculate the derivative, or the so called gradient of the MSE

$$ J'(m, b) = \begin{bmatrix} \frac{df}{dw} \\ \frac{df}{db} \end{bmatrix} = \begin{bmatrix} \frac{1}{N} \sum_{i=1}^{n}{(-2x_i(y_i - (wx_i + b))} \\ \frac{1}{N} \sum_{i=1}^{n}{-2(y_i - (wx_i + b))} \end{bmatrix} $$

And to do this optimization we use the technique called Gradient Descent. We keep altering the weight in the said direction until the MSE approaches Glocal Cost minimum $$ J_{min}(w) $$

Once we have it, we multiply it with the learning rate and subtract it from the (weight, bais) or the parameterst

$$ w = w - \alpha \cdot dw $$
$$ b = b - \alpha \cdot db $$

Learning rate is the deciding factor for how fast or slow to go in the Minimum MSE direction.
A good learning rate will reach the minimum MSE nicely.
A Too low rate will never reach the min MSE, while a too high rate will just keep bouncing off the graph overshooting and creating complete chaos.

#### **Steps**:

- **Training**:
  - Initialize weights as zeros (or prefer random values)
  - Initialize biases as zeros (or prefer random values)
- **Given a data point**:
  - Predict result by using $\hat{y} = wx + b$
  - Calculate error
  - Use gradient descent to figure out new weight and bias
  - Repeat n times

### 3. Logistic Regression

Unline linear regression, we find a probability dstribution map, using the logistic function, generally the sigmoid function

$$ s(x) = /frac{1}{1 + e^{-x}} $$

So the regression equation becomes

$$ \hat{y} = h_{\theta}(x) = /frac{1}{1 + e^{-(wx+b)}} $$

We calculate the error not using MSE, but the Cross Entropy. It is still the derivative of the regression equation.
Gradient Descent is used for adjustments, which calculates the derivates wrt weight and bias. Same equations!

#### **Steps**:

- **Training**:
  - Initialize weights as zeros (or prefer random values)
  - Initialize biases as zeros (or prefer random values)
- **Given a data point**:
  - Predict result by using $\hat{y} = \frac{1}{1 + e^{-(wx + b)}}$
  - Calculate error
  - Use gradient descent to figure out new weight and bias values
  - Repeat n times
- **Testing**
  - Given a data point:
    - Put the values from the data point into the equation $\hat{y} = \frac{1}{1 + e^{-wx + b}}$
    - Choose label based on probability
