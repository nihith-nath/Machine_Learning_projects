import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import cartopy.crs as ccrs
import math

# Problem 1
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True).frame
california_housing.head()
calidata = california_housing
calidata.describe()
calidata.isnull().sum()
calidata.info()
# Box plots



column_names = calidata.columns.tolist()
unuseful_cols = ['Latitude', 'Longitude']
final_cols = [col for col in column_names if col not in unuseful_cols]
print(final_cols)

sns.set(style="whitegrid")


num_cols = len(final_cols)
num_rows = math.ceil(num_cols / 4)


palette = sns.color_palette("hls", num_cols)


fig, axes = plt.subplots(num_rows, 4, figsize=(24, 6*num_rows))
for i, col in enumerate(final_cols):
    row_index = i // 4
    col_index = i % 4
    sns.boxplot(y=calidata[col], ax=axes[row_index, col_index], palette=[palette[i]])
    axes[row_index, col_index].set_title('Box plot of ' + col)
    axes[row_index, col_index].set_ylabel(col)
    axes[row_index, col_index].set_xlabel('Values')


for i in range(num_cols, num_rows*4):
    row_index = i // 4
    col_index = i % 4
    fig.delaxes(axes[row_index, col_index])

plt.tight_layout()
plt.show()

# Histograms


sns.set(style="whitegrid")


sns.set(style="whitegrid")


num_cols = len(final_cols)
num_rows = math.ceil(num_cols / 4)


fig, axes = plt.subplots(num_rows, 4, figsize=(24, 6*num_rows))
for i, col in enumerate(final_cols):
    row_index = i // 4
    col_index = i % 4
    sns.histplot(calidata[col], bins=20, kde=True, ax=axes[row_index, col_index])
    axes[row_index, col_index].set_title('Histogram of ' + col)
    axes[row_index, col_index].set_xlabel('Values')
    axes[row_index, col_index].set_ylabel('Frequency')


for i in range(num_cols, num_rows*4):
    row_index = i // 4
    col_index = i % 4
    fig.delaxes(axes[row_index, col_index])

plt.tight_layout()
plt.show()

#############################
# Scatter plots



min_latitude = calidata['Latitude'].min()
max_latitude = calidata['Latitude'].max()
min_longitude = calidata['Longitude'].min()
max_longitude = calidata['Longitude'].max()


avg_latitude = calidata['Latitude'].mean()
avg_longitude = calidata['Longitude'].mean()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([min_longitude, max_longitude, min_latitude, max_latitude])

sc = ax.scatter(calidata['Longitude'], calidata['Latitude'], c=calidata['MedHouseVal'], cmap='viridis', alpha=0.5)


plt.colorbar(sc, label='Median House Value')


plt.title('Scatter Plot of Latitude vs Longitude on Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


ax.coastlines()
ax.gridlines(draw_labels=True)

plt.show()
############################



min_latitude = calidata['Latitude'].min()
max_latitude = calidata['Latitude'].max()
min_longitude = calidata['Longitude'].min()
max_longitude = calidata['Longitude'].max()


avg_latitude = calidata['Latitude'].mean()
avg_longitude = calidata['Longitude'].mean()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([min_longitude, max_longitude, min_latitude, max_latitude])


sc = ax.scatter(calidata['Longitude'], calidata['Latitude'], c=calidata['MedInc'], cmap='viridis', alpha=0.5)


plt.colorbar(sc, label='Median Income Value')


plt.title('Scatter Plot of Latitude vs Longitude on Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


ax.coastlines()
ax.gridlines(draw_labels=True)

plt.show()



# Calculate the correlation matrix
correlation_matrix = calidata.corr()

# Create a pair plot
sns.pairplot(calidata)
plt.title('Pair Plot of All Columns')
plt.show()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of All Columns')
plt.show()

# Problem 2 Linear Regression


class LinearRegression:
    def _init_(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.m = None
        self.bias = None

    def fit(self, X, y):
        n_samples = len(X)
        self.m = 0
        self.bias = 0

        X = X.tolist()
        y = y.tolist()

        for _ in range(self.n_iterations):
            y_predicted = [self.m * x_i + self.bias for x_i in X]

            dm = (1 / n_samples) * sum([(y_predicted[i] - y[i]) * X[i] for i in range(n_samples)])
            db = (1 / n_samples) * sum([y_predicted[i] - y[i] for i in range(n_samples)])

            self.m -= self.learning_rate * dm
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return [self.m * x + self.bias for x in X]

california_housing = fetch_california_housing(as_frame=True)

calidata = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
calidata['MedHouseVal'] = california_housing.target

X = calidata[['MedInc']]
y = calidata['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=420)

model = LinearRegression()
model.fit(X_train['MedInc'], y_train)

predictions = model.predict(X_test['MedInc'])

print("Predictions:", predictions[:5])

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual',s=1)
plt.plot(X_test, predictions, color='red', label='Predicted')
plt.title('Linear Regression Model')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

train_predictions = model.predict(X_train['MedInc'])
test_predictions = model.predict(X_test['MedInc'])

train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train R2 Score:", train_r2)
print("Test R2 Score:", test_r2)

train_errors = y_train - train_predictions
test_errors = y_test - test_predictions

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(train_predictions, train_errors, color='blue', label='Train Data',s=1)
plt.title('Errors on Train Data')
plt.xlabel('Predicted Values')
plt.ylabel('Errors')
plt.legend()
plt.axhline(y=0, color='black', linestyle='--')

plt.subplot(1, 2, 2)
plt.scatter(test_predictions, test_errors, color='red', label='Test Data',s=1)
plt.title('Errors on Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Errors')
plt.legend()
plt.axhline(y=0, color='black', linestyle='--')

plt.tight_layout()
plt.show()

# Problem 3 Multiple Linear Regression

X = calidata.iloc[:,:-1].values
y = calidata['MedHouseVal'].values

def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

X_train_normalized, mean, std = normalize_features(X_train)
X_test_normalized = (X_test - mean) / std

n_samples, n_features = X_train_normalized.shape
weights = np.zeros(n_features)
bias = 0.0
learning_rate = 0.01
iterations = 1000

for _ in range(iterations):
    y_pred = np.dot(X_train_normalized, weights) + bias
    dw = (1 / n_samples) * np.dot(X_train_normalized.T, (y_pred - y_train))
    db = (1 / n_samples) * np.sum(y_pred - y_train)
    weights -= learning_rate * dw
    bias -= learning_rate * db

y_pred_test = np.dot(X_test_normalized, weights) + bias
y_pred_train = np.dot(X_train_normalized, weights) + bias

mse_train = np.mean((y_pred_train - y_train) ** 2)
mse_test = np.mean((y_pred_test - y_test) ** 2)
y_mean = np.mean(y_test)
ss_tot = np.sum((y_test - y_mean) ** 2)
ss_res = np.sum((y_test - y_pred_test) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)
print("R-squared:", r_squared)
print("weights:", weights)
print("bias:", bias)

train_errors = y_train - (np.dot(X_train_normalized, weights) + bias)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_train, train_errors, color='blue',s=1)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Errors on Train Data')
plt.xlabel('Predited Values')
plt.ylabel('Errors')
plt.show()

test_errors = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, test_errors, color='green',s=1)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Errors on Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Errors')
plt.show()
# Problem 4




california_housing = fetch_california_housing(as_frame=True)



calidata = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
calidata['MedHouseVal'] = california_housing.target


calidata_subset = calidata.sample(frac=0.2, random_state=42)
X = calidata_subset.drop(columns=['MedHouseVal', 'Population', 'AveOccup', 'AveBedrms'])
y = calidata_subset['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=420)


def gaussian_kernel(xq, xi, tau):
    m = len(xi)
    weights_matrix = np.zeros((m, m))
    for j in range(m):
        weight = xq - xi[j]
        weights_matrix[j, j] = np.exp(-np.dot(weight, weight) / (2 * tau**2))
    return weights_matrix


def local_weight(xq, xi, yi, tau):
    m = len(xi)
    xi_augmented = np.hstack((np.ones((m, 1)), xi))  
    xq_augmented = np.append([1], xq)  
    weights = gaussian_kernel(xq_augmented, xi_augmented, tau)
    x_weights = np.dot(xi_augmented.T, np.dot(weights, xi_augmented))
    y_weights = np.dot(xi_augmented.T, np.dot(weights, yi.reshape(-1, 1)))
    theta = np.linalg.inv(x_weights).dot(y_weights)
    return theta


def local_weight_regression_adjusted(xi, yi, xq, tau):
    y_pred_test = np.zeros(len(xq))
    for i in range(len(xq)):
        theta = local_weight(xq[i], xi, yi, tau)
        y_pred_test[i] = np.dot(np.append([1], xq[i]), theta).flatten()
    return y_pred_test


X_train_single_feature = X_train[['MedInc']].values
X_test_single_feature = X_test[['MedInc']].values


tau_values = [0.1, 0.5, 1, 10]

def plot_lwlr_lines(X_train, y_train, X_test, y_test, tau_values):
    plt.figure(figsize=(20, 10))

    for tau in tau_values:
        y_pred_train = local_weight_regression_adjusted(X_train_single_feature, y_train, X_train_single_feature, tau)
        y_pred_test = local_weight_regression_adjusted(X_train_single_feature, y_train, X_test_single_feature, tau)

        sort_index = np.argsort(X_train_single_feature[:,0])
        X_train_sorted = X_train_single_feature[sort_index,0]
        y_pred_train_sorted = y_pred_train[sort_index]

        plt.plot(X_train_sorted, y_pred_train_sorted, label=f'Tau={tau}')

    plt.scatter(X_train_single_feature[:,0], y_train, color='black', label='Actual Data', alpha=0.5)
    plt.title('LWLR Model Lines for Different Tau Values')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.legend()
    plt.show()

plot_lwlr_lines(X_train_single_feature, y_train.values, X_test_single_feature, y_test.values, tau_values)

def evaluate_model(X_train, y_train, X_test, y_test, tau_values):
    train_mse_values = []
    test_mse_values = []
    train_r2_values = []
    test_r2_values = []

    for tau in tau_values:
        y_pred_train = local_weight_regression_adjusted(X_train_single_feature, y_train, X_train_single_feature, tau)
        y_pred_test = local_weight_regression_adjusted(X_train_single_feature, y_train, X_test_single_feature, tau)

        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        train_mse_values.append(train_mse)
        test_mse_values.append(test_mse)
        train_r2_values.append(train_r2)
        test_r2_values.append(test_r2)

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.scatter(y_pred_train, y_train - y_pred_train, color='blue', label='Train Data', s=1)
        plt.scatter(y_pred_test, y_test - y_pred_test, color='green', label='Test Data', s=1)
        plt.xlabel('Predicted Values')
        plt.ylabel('Errors')
        plt.title('Residual Plot for Tau={}'.format(tau))
        plt.legend()

        

        plt.show()

    return train_mse_values, test_mse_values, train_r2_values, test_r2_values

train_mse_values, test_mse_values, train_r2_values, test_r2_values = evaluate_model(X_train_single_feature, y_train.values, X_test_single_feature, y_test.values, tau_values)

for i, tau in enumerate(tau_values):
    print(f'Tau={tau}: Train MSE={train_mse_values[i]}, Test MSE={test_mse_values[i]}, Train R^2={train_r2_values[i]}, Test R^2={test_r2_values[i]}')


# Problem 5
    # Code for model evaluation is included in respective models