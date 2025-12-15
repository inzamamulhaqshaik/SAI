# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

# Step 3: Explore the Data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Step 4: Visualize the Data
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(df['distance_km'], df['delivery_time_min'],
                           c=df['delivery_time_min'], cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[0].set_xlabel('Distance in Kms', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Delivery Time in Min', fontsize=12, fontweight='bold')
axes[0].set_title(' Distance in Kms vs Delivery Time in Min', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter1, ax=axes[0], label='Delivery Time (ETA)')

scatter2 = axes[1].scatter(df['prep_time_min'], df['delivery_time_min'],
                           c=df['delivery_time_min'], cmap='plasma', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[1].set_xlabel('Prep Time in Mins', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Delivery Time in Min', fontsize=12, fontweight='bold')
axes[1].set_title('Prep Time in Mins vs Delivery Time in Min', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter2, ax=axes[1], label='Delivery Time (ETA)')

plt.tight_layout()
plt.show()

# Step 5: Prepare Data for Training
X = df[['distance_km', 'prep_time_min']]  # Two features now
y = df['delivery_time_min']

# Step 6: Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Check Model Parameters
print("\nModel Parameters:")
print(f"Coefficients: distance_km = {model.coef_[0]:.2f}, prep_time_min = {model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Step 9: Make Predictions on Test Data
y_pred = model.predict(X_test)

print("\nPredictions vs Actual:")
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred.round()})
print(results)

# Step 10: Model Accuracy
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(f"\nModel Accuracy (R² Score): {score:.2f}")

# Step 11: Answer Priya's Question
new_project = [[10, 5]]  # 10 distance_km, 5 days deadline
predicted_rate = model.predict(new_project)
print(f"\nVikram's question — for 7 km distance and 15 min prep time, what is the expected delivery time?")
print(f"Recommended Rate: ₹{predicted_rate[0]:,.0f}")