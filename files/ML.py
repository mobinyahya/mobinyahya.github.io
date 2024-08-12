import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error






################################################################ Pandas
#######################################################################
data = {'Name': ['Tom', 'Nick', 'Anna', 'Mo', 'Jon', 'Mike', 'Arec',  'Ali'],
        'Zipcode' : [193420, 234518, 412354, 992354, 412354, 124104, 112354, 488854],
        'Married': [0, 0, 1, 1, 1, 0, 0, 1],
        'Kid': [0, 0, 3, 8, 1, 0, 2, 0],
        'Age': [19, 21, 35, 71, 65, 12, 25, 21],
        'Salary': [100, 120, 110, 200, 180, 50, 130, 112]}

df = pd.read_csv("Mobin/Downloads")
# Create DataFrame
df = pd.DataFrame(data)
df1 = df2 = df

df = df.fillna(0)
df_teenage = df[df['Age'] < 22]
df_Age21s = df[df["Age"] == 21]

# Sort the df in descending order
df_sorted = df.sort_values(by='Age', ascending=False)


# Row lambda operation on a single column
df['Age_Double'] = df['Age'].apply(lambda x: x * 2)

df = df.drop('Age_Double', axis=1)

# Row lambda operation on the whole row
df["Filter"] = df.apply(lambda row: 1 if "Tom" not in row["Name"] else 0, axis=1)
df = df.loc[df["Filter"] == 1]

df.reset_index(inplace=True) # Add drop=True, to drop the old index column

df.loc[df['Zipcode'] < 234518, 'Salary'] = 100

row_number_2 = df.iloc[[2]]
rows_1_2_3 = df[1:4]


df = df.groupby("Zipcode").sum()
df = df1.merge(df2, how="inner", on="Zipcode")
column_names = df.columns
print(column_names)
df.rename(columns={'Name': 'First_Name'}, inplace=True)
df = df[['Zipcode', 'Age']] 




for idx, row in df.iterrows():
    df["Zipcode"][idx] = 0
    if (row["Age"] < 0):
        print("")


df.to_csv('my_dataframe.csv')

################################################################# Numpy
#######################################################################
arr = np.array([1, 2, 3]) # an array with given values


arr = np.linspace(0, 10, 11) # arr = [0,1,2,...,10]
arr = np.zeros((3,4)) # 3 * 4 array of zeros
arr = np.ones((3,4)) # 3 * 4 array of ones

# single random number between 0 and 1
random_number = np.random.rand()
# 2x3 random array
random_array = np.random.rand(2, 3)
# 2x3 array of random integers between 1 and 10
random_integers = np.random.randint(1, 10, size=(2, 3))
# 2x3 array with samples from a standard normal distribution
random_norm = np.random.randn(2, 3)


row_0 = arr[0]
column_0 = arr[:,0]
sub = arr[0:2, 2:] # rows 0,1 columns 2,3,.. end

shape = arr.shape # = tuple (3,4)
ndim = arr.ndim # = 2

mean = arr.mean() # or sum, max, min, std, var,

A = B = arr

# Element-Wise product, iff same shape
product = A *  B # or subtract (-), sum (+), divide (/)

transpose = arr.T

A = B = arr

combined_vertical = np.concatenate((A, B), axis=0) #axis=0: first dimension of arrays

dot_product = np.dot(A, B) #dot product of two ve

matrix_product = A @ B

determinant = np.linalg.det(A)
inverse = np.linalg.inv(A)
eigenvalues, eigenvectors = np.linalg.eig(A)
U, S, V = np.linalg.svd(A, full_matrices=True)

b = np.array([9, 8])
x = np.linalg.solve(A, b) # solve for Ax = b


################################################## Quick Classification
#######################################################################

# -------------------- Feature Stats
# -----------------------------------
con_cols = ['Kid']
print(data.describe())
print(data.info())
(df.corr())
# Show Missing values
print(df.isnull().sum())


# Use one-hot-encoding on features Name and Zipcode
df_encoded = pd.get_dummies(df, columns=['Name', 'Zipcode'])

# Separate X and y values
X = df.drop(columns=["Salary"])
y = df[["Salary"]]


# Split the test-train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# -------------------- Feature Scaling
# -----------------------------------
#If SGD: Scale features to have mean 0, variance 1. Returns numpy array
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# or instead: 
X_train[con_cols] = scaler.fit_transform(X_train[con_cols])
X_test[con_cols] = scaler.transform(X_test[con_cols])


# Define which model you want to use
model = RandomForestClassifier()
model = LinearRegression() # compute using direct matrix multiplication
regressor = SGDRegressor(max_iter=1000,  # Number of epochs
                         eta0=0.01,      #  Initial learning rate
                         learning_rate='constant')  # How the learning rate changes over time (across epochs). Options: 'constant', 'optimal', 'invscaling', 'adaptive'



# Train the model
model.fit(X_train, y_train)

# Show feature importance
feature_selection = False
if feature_selection:

    # --------- Approach in RandomForest
    # ------------------------------------
    model.fit(X_train, y_train)
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    # ------- Approach in LinearRegression
    # ------------------------------------
    # Recursive Feature Elimination
    rfe = RFE(model, n_features_to_select=3)
    rfe.fit(X_train, y_train)

    # Print the boolean mask of selected features
    print(f"Selected features: {X.columns[rfe.support_]}")

    # Print the ranking of the features
    print(f"Feature ranking: {rfe.ranking_}")


    # Transform the data to only include the selected features
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)

    # Fit the model again using the selected features
    model.fit(X_train_selected, y_train)

# ------------------------------------
# ---------------------- Lasso / Ridge
# ------------------------------------
# alpha: regularization strength
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)

# Fit model
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

# Lasso shrinks coefficients to zero. Find non-zero ones
selected_features = X.columns[(lasso.coef_ != 0)]



# ------------------------------------
# Predict on test data
y_pred = model.predict(X_test)

# Mean squared error
mse_loss = mean_squared_error(y_test, y_pred)
# Mean squared error computed manually
y_test = y_test.to_numpy()
mse_loss = np.mean((y_test - y_pred) ** 2)

# Evaluation on test data - for classification problems
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# ------------------------------------
# ----------- k-fold cross-validation
# ------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation
# TODO can you pass the x_test (numpy) to this instead of the X dataframe?
#  TODO should I put cv = kf, or cv = 5 (similar to the juypyter project)
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

# scores are negative (neg_mean_squared_error), Convert to positive
mse_scores = -scores

print(f"Mean Squared Error for each fold: {mse_scores}")
print(f"Average Mean Squared Error: {mse_scores.mean()}")


############################################################  Plotting
#######################################################################
# Plotting y_pred vs y_test
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Age'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Age'], y_pred, color='red', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()


