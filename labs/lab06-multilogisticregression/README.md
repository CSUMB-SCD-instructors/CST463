#Lab: Logistic Regression (multi-class)

Hints for problems are provided at the end of the lab.  Refer to hints only as needed, or to check your answer.

# Q1
Explain, in your own words, how to implement multi-class logistic regression with gradient descent.  Write it down.

# Q2
Write down how a one-vs-all classifier works.

# Q3
How many binary classifiers do you need to make a one-vs-all classifier when there are K classes?

# Q4
How many binary classifiers do you need to make a one-vs-one classifier when there are K classes?

# Q5
Using NumPy, write a function that will give cross-entropy loss for an array of class probability and a one-hot-encoded target value.  Your code could be called like this:
```python
cross_entropy(np.array([0.1, 0.7, 0.2]), np.array([0, 1, 0]))
```

This would be for a 3-class classification problem.  Test that your function is correct.  See examples from the lecture.

# Q6
If you still have time, extend your function so that it can take a 2D array of predictions (one prediction per row) and a 2D array of 1-hot encoded target values (one per row) and return the average cross entropy for the collection of predictions/target values.
