# Lab: Gradient Descent for linear regression

## Questions

### Q1

Generate data values `X`, `X_b` and `y` using the code from Geron below.
Read the code with your team and make sure you fully understand what it's doing.

```python

m = 100
theta = np.array([4, 3])	   # y intercept = 4, slope = 3
X = 2 * np.random.rand(m, 1)
y = theta[0] + theta[1] * X + np.random.randn(m, 1)  
X_b = np.c_[np.ones((m, 1)), X]  # augmented version of X


```

<details><summary>Hint</summary>

This code is generating data that lies close to the line y = 4 + 3x.

</details>

### Q2

Studey Geron's code shown below for batch gradient descent and turn it into a function name `batch_gradient_descent`.  
Make sure you understand the starting code.

Don't forget to return the final result!

```python

eta = 0.1
n_iterations = 1000
theta = np.random.randn(2,1)     # initialize randomly

for iteration in range(n_iterations):
  gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)   
  theta = theta - eta*gradients


```

```python

def batch_gradient_descent(...):
  # todo

```



<details><summary>Hint</summary>

Your function definition might look like this:

```python

def batch_gradient_descent(X_b, y, eta=0.1, n_iterations=100):
  # body of your function definition here

```


</details>

### Q3

Run your code on the data from problem 1.
Did you get the right answer?

Try running the code for a small number of iterations and print out the gradient values at each iteration.

### Q4

Modify the function again so that it works with code with any number of features.
Does `theta` have to change?

<details><summary>Hint</summary>

Yes, the number of rows in theta has to match the number of columns in `X_b`

</details>

### Q5

Modify the algorithm so that it keeps a history of the thetas across iterations and return both `theta` and `theta` hsitory, where `theta_history` is a matrix.

<details><summary>Hint</summary>

You could define a variable `theta_history` intilized to `np.zeros((n_iterations, n))`

</details>


### Q6

Run the algorithm and plot the history of theta as a scatter plot.

<details><summary>Hint</summary>

Your history should have two columns.
Yse a scatter plot, plot the first column as x and the second column as y.

</details>

### Q7

Can you explain why the history is shaped as it is?


<details><summary>Hint</summary>

If there's a sharp bend in the plot, this normally indicates that the data is not scaled.

</details>

### Q8

Scale the data (only X!) and repeat step 6