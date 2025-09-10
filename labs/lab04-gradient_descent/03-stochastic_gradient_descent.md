# Lab: Stochastic Gradient Descent

## Q1

Write a function to get a batch from some data.  The function should look like this, where `X` and `y` are NumPy arrays, and `batch_size` is an integer.

```python
def make_batch(X, y, batch_size):
  # YOUR CODE HERE
  return X_batch, y_batch
```

## Q2

Generate data values X, X_b, and y (this code is based on Geron’s code on p. 108).  
The idea is to generate points on the line defined by theta[0] and theta[1], and then to add noise.

```python
np.random.seed(0)
m = 200
theta = np.array([4, 3])	   # y intersect = 4, slope = 3
X = 10 * np.random.rand(m, 1)
y = theta[0] + theta[1] * X
y = y + np.random.randn(m, 1)    # add noise
X_b = np.c_[np.ones((m, 1)), X]  # augmented version of X 
```

## Q3

Study Geron’s code for batch gradient descent, and turn it into a function named `batch_gradient_descent`.  Add parameters for `eta` and `n_iterations`.  
Here’s Geron’s original code:

```python
eta = 0.01
n_iterations = 1000
theta = np.random.randn(2,1)     # initialize randomly

for iteration in range(n_iterations):
  gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)   
  theta = theta - eta*gradients
```

You will want to return the final result.
Test your function and see if you something close to `[4, 3]` as the result.  You will probably need to tweak `eta` and `n_iterations`.

## Q4

Copy your `batch_gradient_descent()` function and rename it `minibatch_gradient_descent()`.  
Change the code so that mini-batch gradient descent is used.  
You will want to add a `batch_size` parameter.  
Use a default batch size of 10.

Test your function.

## Q5 Modify your algorithm so that it returns the history of theta values.  For example, return a list of the theta arrays.  

Run the algorithm, then and plot the history of theta as a scatter plot.  
You'll have to turn your history of theta values into an array containing the first theta value and an array containing the second theta value.  
I suggest using `plt.xlim()` and` plt.ylim()`.


## Q6

Repeat the previous step, but with a different batch size.  Use the same `plt.xlim()` and `plt.ylim()` values as before to make comparison easier.


## Q7

Repeat the previous step with a couple more batch sizes.
