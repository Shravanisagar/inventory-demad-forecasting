import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Preprocessing
# Separate features and target variable
X = data[['price', 'revenue']]  # Independent variables
y = data['demandPrediction']    # Dependent variable

# Step 2: Visualizations
# Histogram of demandPrediction
plt.hist(y, bins=20, color='skyblue', alpha=0.7)
plt.title('Histogram of Demand Prediction')
plt.xlabel('Demand Prediction')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 5: Prediction and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 6: Display Results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared Score: {r2:.2f}")

# Step 7: Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.title('Actual vs Predicted Demand Prediction')
plt.xlabel('Actual Demand Prediction')
plt.ylabel('Predicted Demand Prediction')
plt.show()