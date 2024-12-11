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

Find the implementation in the file `linear_regression.py`

### 3. Logistic Regression

Unline linear regression, we find a probability dstribution map, using the logistic function, generally the sigmoid function

$$ s(x) = \frac{1}{1 + e^{-x}} $$

So the regression equation becomes

$$ \hat{y} = h_{\theta}(x) = \frac{1}{1 + e^{-(wx+b)}} $$

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
    - Put the values from the data point into the equation $\hat{y} = \frac{1}{1 + e^{-(wx + b)}}$
    - Choose label based on probability

Find the implementation in the file `logistic_regression.py`

### 4. Decision Trees

Decision trees are hierarchical structures that split the data into subsets based on feature values.
Starting from a root node containing all data, the algorithm recursively partitions the data by choosing optimal splitting criteria at each node.
The goal is to create homogeneous groups in the leaf nodes.
This makes the prediction data to trickle down the tree answer each splitting question, into it's catergory/classification

We need to decide on the
1. Split feature
2. Split point
3. When to stop splitting

#### **Steps**:

- **Training**:
  Given the whole dataset:
    - Calculate _information gain_ with each possible split
    - Divide set with that feature and value that gives the most IG
    - Divide tree and do the same for all created branches...
    - ...until a _stopping criteria_ is reached

- **Testing**:
  Given a data point:
    - Follow the tree until you reach a leaf node
    - Return the most common class label

**Information Gain**: $IG = E(parent) - [weighted\  average] \cdot E(children)$

where E(x) is entropy, lack of order.
Entropy of a node is zero is there is only one class in it.
Entropy is 1 if classes are evenly spread in a node

**Entropy**: $E = - \sum{p(X) \cdot log_2(p(X))}$

Where, $p(X) = \frac{\#x}{n}$, i.e. the number of times class in node / number of total nodes

**Stopping Criteria**: Maximum depth, minimum number of samples, min impurity decrease.

Find the implementation in the file `decision_tree.py`


### 5. Random Forest

Random Forest creates a specified number of trees (naturally)! on a random subset of samples, and predicts based on the majority vote of all trees.

#### **Steps**:

- **Training**:
  Given the whole dataset:
    - Get a subset of the dataset randomly
    - Create a decision tree
    - Repeat for as many times as the number of trees

- **Testing**:
  Given a data point:
    - Get the predictions from each tree
    - _Classificiation_: hold a majority vote
    - _Regression_: get the mean of the precitions

Find the implementation in the file `random_forest.py`


### 6. Naive Bayes

The Naive Bayes classifier is a "probabilistic classifier" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

Bayes' Theorem states
$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$

Translating to our case, we have

$$ P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)} $$
Where, y are the class labels while X is the feature vector.

Now for this, the features must be mutually independent.
$$ P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)} $$
$$ \downarrow $$
$$ P(y|X) = \frac{P(x_1|y) \cdot P(x_2|y) \cdot P(x_3|y) \cdot \dots \cdot P(y)}{P(X)} $$

Select class with highest posterior porbability (that $P(y|X)$ )

$$ y = argmax_y P(y|X) $$
since it contains product terms, and each probability is between 0 and 1, multiplication can result in exteml
$$ y = argmax_y log(P(x_1|y)) + log(P(x_2|y) + ... + log(P(x_n|y)) + log(P(y)) $$

#### Prior and Class Conditional

$P(y)$ - Prior probability --> Frequency of each class
$P(x_i|y)$ - Class conditional probability --> Model with Gaussian

Gaussian Formula
$$ P(x_i|y) = \frac{1}{\sqrt{2 \pi \sigma_y^2}} \cdot exp(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}) $$

#### **Steps**:

- **Training:**
  - Calculate mean, var, and prior (frequency) for each class

- **Predictions:**
  - Calculate posterior for each class with $y = argmax_y log(P(x_1|y)) + log(P(x_2|y) + ... + log(P(x_n|y)) + log(P(y))$ and the Gaussian formula
  - Choose class with highest posterior probability


Find the implementation in the file `naive_bayes.py`
