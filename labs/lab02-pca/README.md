# Lab: PCA Fitting

## Instructions

Try to answer from memory and only use the hints when you absolutely need to.

## Questions

### Q1

Name the general approaches to dimensionality reduction that we discussed

<details><summary>Answer</summary>

PCA and SVD

</details>

### Q2

In PCA, why does it make sense to look for the axis along which there is the most variance?

<details><summary>Answer</summary>

Since it has the most variance, this axis explains most of the variation in the data and is therefore going to lead differentiate individual data points more effectively.

</details>

### Q3

(True/False)
After applying PCA to transform a data set, each feature in an instance of the new data is the linear combination of features in teh corresponding instance of the original data.

<details><summary>Answer</summary>

True, since we are just restating the original data in a different coordinate system.

</details>


### Q4

Please complete the code found in [pca.py](pca.py).
