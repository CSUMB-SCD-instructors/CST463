# Lab: Gradient Descent

## Questions

### Q1

From memory, write out the pseudocode for gradient descent

### Q2

Using the function $` f(x) = -(x-1)^2 + 2 `$ answer the following questions

#### Q2a

Define the function in python

<details><summary>Answer</summary>

```python

def f(x):
  return -1 * (x - 2)**2 + 2

```

</details>

#### Q2b

Plot the function $`f(x)`$ from -1 to 3

#### Q2c

Looking at your plot, approximately what is the value of x that maximizes $`f(x)`$

#### Q2d

What is $` \frac{df}{dx} `$?

<details><summary>Answer</summary>

$` \frac{df}{dx} = -2x + 2`$?

</details>


#### Q2e

Starting with x= 0 0, perform a step of gradient descent using a learning rate of 0.1
Use your python console if you want.

#### Q2f

Perform a few more steps of gradient descent by hand.

#### Q2g

Repeat step 6 but with a learning rate of 0.5

#### Q2h

How would your process change if you wanted to find the minimum of a function?