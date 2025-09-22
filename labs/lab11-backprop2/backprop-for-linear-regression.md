# Lab: BackProp for linear regression

Work closely with your team; compare your answers and resolve any differences.

## Q1
Refer to slide 2 of the lecture.  First, write the equations for the network.

## Q2
We will repeat the work of the last lab, except this time with linear regression.  Use these values:

```math
\begin{matrix}
x = (1, 0.5, 2)   \\
y = 0             \\
b = (0.1, 1, 0.5) \\
\end{matrix}
```

## Q3
Perform backprop step by step to find the partial derivatives shown on the slide.  This time, write code to perform the calculations.  Make sure you get the same answer as your teammates.

## Q4
Using your result from step 3, and using learning rate $`\alpha = 0.1`$, compute the updated value of coefficients b.

## Q5
Here is some code to generate data:
```python
# true parameter values
b0_true = 0.5
b1_true = -0.5
b2_true = 1.5

# generate some random training data
m = 100
x1_all = np.random.uniform(0, 10, size=m)
x2_all = np.random.uniform(0, 10, size=m)
y_all = (b0_true + b1_true*x1_all + b2_true*x2_all) + np.random.normal(scale=0.25, size=m)
```

Generate the data, then run gradient descent to find the values of b0, b1, b2.  
You may need to tweak the learning rate and number of iterations. 

## Q6
If you still have time (I don't think you will), using the diagram below, perform one step of gradient descent for the neuron shown in the diagram below.  Your goal is to compute the new values of w1, w1.  
