# Lab: Backprop preparation

Work closely with your team; compare your answers and resolve any differences.

## Q1
Here are the equations for the logistic regression network in lecture:

```math

\begin{matrix}
u(\beta) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 \\
v(u) = \sigma(u) \\
z(v) = -(y log(v) + (1-y) log(1-v))
\end{matrix}

```
 
$\sigma(u)$ is the sigmoid function.  
Work out the partial derivatives of z with respect to each of the $\beta$ variables.  
Use the multivariable chain rules and these basic derivatives:

- if $`y = \sigma(x)`$, then $`\frac{\delta y}{\delta x} = \sigma(x)(1 - \sigma(x))`$
- if $`y = log(x)`$, then   $`\frac{\delta y}{\delta x} = \frac{1}{x}`$
- if $`y = log(1-x)`$, then $`\frac{\delta y}{\delta x} = \frac{-1}{1-x}`$

The last of these rules can be derived using the chain rule.

## Q2

If you still have time, check your work by comparing what you got with what you would get if you estimated the derivative numerically.  
For example, for $\frac{\delta z}{\delta v}$, you would want to evaluate your expression at a certain $y$ and $v$.  
Then compute it numerically by evaluating the equation for $z$ twice, at two values very close to $v$.  
This will give you an estimate of the partial derivative.
