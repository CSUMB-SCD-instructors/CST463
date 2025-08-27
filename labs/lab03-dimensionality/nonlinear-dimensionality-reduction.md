# Lab: Non-linear dimensionality reduction



## Questions

### Q1

True/False.  If you use PCA but don't reduce the number of dimensions, the total variance in the transformed space is the same as the total variance in the original space.

<details><summary>Answer</summary>

True.  
We saw that the benefit of PCA is that it tries to capture as much variance as possible along the first dimension in the new space.  But if you use all dimensions in the new space, all points will be captured just as precisely as in the original space.

</details>


### Q2

Compare plain PCA with LLE and kernel PCA.  Here is some starter code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#
# read the red wine data
#

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst495/master/winequality-red.csv", sep=";")

# any NaN values?
df.isnull().sum()

X = df.iloc[:,0:11]
y = df['quality'].values

#
# Use linear regression on the original data.
# 
# split the data into training and test sets, train
# a linear model, make predictions, and compute RMSE
#

X_train, X_test, y_train, y_test = train_test_split(X,y)

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)


#
# (a) fit PCA to the data
#

# YOUR CODE HERE

#
# (b) transform X to a new data set using the top three principal components
#
# The best way to do this is to fit on the training data and then use the
# fitted PCA object to transform both the training and test data.
#

# YOUR CODE HERE

#
# (c) try linear regression on the transformed data created
# with the first three principal components.  Compare
# your RMSE to the previous case.
# 

# YOUR CODE HERE

#
# (d) Repeat steps 2 and 3 using LLE
# 

# YOUR CODE HERE

#
# (e) Repeat steps 2 and 3 using kernel PCA
# 

# YOUR CODE HERE

```

