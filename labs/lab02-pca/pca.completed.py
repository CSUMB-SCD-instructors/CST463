import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#
# read the red wine data
#

df = pd.read_csv("winequality-red.csv", sep=";")

# any NaN values?
df.isnull().sum()

X = df.iloc[:,0:11]
y = df['quality'].values

#
# use PCA and plot the cumulative explained variance ratio
#

# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()

# Print explained variance ratios
print("Explained variance ratio for each component:")
for i, ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

print(f"\nCumulative explained variance with 3 components: {cumulative_variance_ratio[2]:.4f}")

#
# transform X to a new data set using the top three principal components
#

# Fit PCA with 3 components
pca_3 = PCA(n_components=3)
X_pca = pca_3.fit_transform(X)

print(f"\nOriginal data shape: {X.shape}")
print(f"Transformed data shape: {X_pca.shape}")

#
# try linear regression on the original data
#
# split the data into training and test sets, train
# a linear model, make predictions, and compute RMSE

X_train, X_test, y_train, y_test = train_test_split(X,y)

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\nLinear Regression on Original Data:")
print(f"RMSE: {rmse:.4f}")

#
# try linear regression on the transformed data created
# with the first three principal components.  Compare
# your RMSE to the previous case.
#

# Split the PCA-transformed data
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, random_state=42)

# Train linear regression on PCA data
regr_pca = LinearRegression()
regr_pca.fit(X_pca_train, y_pca_train)
y_pca_pred = regr_pca.predict(X_pca_test)
rmse_pca = mean_squared_error(y_pca_test, y_pca_pred, squared=False)

print(f"\nLinear Regression on PCA-transformed Data (3 components):")
print(f"RMSE: {rmse_pca:.4f}")

print(f"\nComparison:")
print(f"Original data RMSE: {rmse:.4f}")
print(f"PCA data RMSE: {rmse_pca:.4f}")
print(f"Difference: {rmse_pca - rmse:.4f}")

if rmse_pca > rmse:
    print("The PCA-transformed data has higher RMSE, indicating some information loss.")
else:
    print("The PCA-transformed data has lower RMSE, indicating good dimensionality reduction.")