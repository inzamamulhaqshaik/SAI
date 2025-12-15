import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16, 4, 8, 4, 16, 8, 8, 4, 16, 8, 16, 4, 8, 4, 16, 8, 8, 4, 16, 8, 16],
    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512, 128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024, 256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],
    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0, 1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6, 2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],
    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000, 20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000, 25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

# ============================================
# 1. LOAD AND EXPLORE THE DATA
# ============================================
print("=" * 60)
print("LAPTOP PRICE PREDICTION MODEL")
print("=" * 60)
print("\n Dataset Overview:")
print(df.head(10))
print(f"\n Dataset Shape: {df.shape[0]} laptops, {df.shape[1]} features")
print("\n Statistical Summary:")
print(df.describe())

# ============================================
# 2. VISUALIZE THE RELATIONSHIPS
# ============================================
print("\n" + "=" * 60)
print("VISUALIZATION: How Each Spec Affects Price")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Laptop Specs vs Price', fontsize=16, fontweight='bold')

# Plot 1: RAM vs Price
axes[0].scatter(df['ram_gb'], df['price_inr'], alpha=0.6, s=100, c='#3498db', edgecolors='black')
axes[0].set_xlabel('RAM (GB)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axes[0].set_title('RAM vs Price', fontsize=13)
axes[0].grid(alpha=0.3)

# Plot 2: Storage vs Price
axes[1].scatter(df['storage_gb'], df['price_inr'], alpha=0.6, s=100, c='#e74c3c', edgecolors='black')
axes[1].set_xlabel('Storage (GB)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axes[1].set_title('Storage vs Price', fontsize=13)
axes[1].grid(alpha=0.3)

# Plot 3: Processor vs Price
axes[2].scatter(df['processor_ghz'], df['price_inr'], alpha=0.6, s=100, c='#2ecc71', edgecolors='black')
axes[2].set_xlabel('Processor Speed (GHz)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axes[2].set_title('Processor vs Price', fontsize=13)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 3. TRAIN LINEAR REGRESSION MODEL
# ============================================
print("\n" + "=" * 60)
print("TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

# Prepare features (X) and target (y)
X = df[['ram_gb', 'storage_gb', 'processor_ghz']]
y = df['price_inr']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print(f" Model trained successfully!")
print(f" Training samples: {len(X_train)}")
print(f" Testing samples: {len(X_test)}")

# ============================================
# 4. CHECK COEFFICIENTS
# ============================================
print("\n" + "=" * 60)
print("MODEL COEFFICIENTS: Which Spec Matters Most?")
print("=" * 60)

coefficients = pd.DataFrame({
    'Feature': ['RAM (GB)', 'Storage (GB)', 'Processor (GHz)'],
    'Coefficient': model.coef_,
    'Impact': [f'₹{coef:,.2f} per unit' for coef in model.coef_]
})

print(coefficients.to_string(index=False))
print(f"\nIntercept (Base Price): ₹{model.intercept_:,.2f}")

# Determine which feature has the most impact
max_coef_idx = np.argmax(np.abs(model.coef_))
feature_names = ['RAM', 'Storage', 'Processor Speed']
print(f"\n Most Important Feature: {feature_names[max_coef_idx]}")
print(f" Impact: ₹{model.coef_[max_coef_idx]:,.2f} per unit increase")

# ============================================
# 5. CALCULATE R² SCORE
# ============================================
print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f" R² Score (Training): {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f" R² Score (Testing):  {r2_test:.4f} ({r2_test*100:.2f}%)")

if r2_test > 0.9:
    print(" Excellent! The model explains the price variation very well!")
elif r2_test > 0.7:
    print(" Good! The model is reasonably accurate.")
else:
    print("  Fair. The model could be improved with more features.")

# ============================================
# 6. ANSWER MEERA'S QUESTION
# ============================================
print("\n" + "=" * 60)
print("MEERA'S QUESTION: Fair Price Prediction")
print("=" * 60)

# Predict for 16GB RAM, 512GB Storage, 3.2 GHz Processor
meera_specs = np.array([[16, 512, 3.2]])
predicted_price = model.predict(meera_specs)[0]

print(f"\n Laptop Specs:")
print(f"   -RAM: 16 GB")
print(f"   -Storage: 512 GB")
print(f"   -Processor: 3.2 GHz")
print(f"\n  Predicted Fair Price: ₹{predicted_price:,.2f}")

# Find similar laptops in the dataset
similar = df[(df['ram_gb'] == 16) & (df['storage_gb'] == 512) & (df['processor_ghz'] == 3.2)]
if not similar.empty:
    actual_price = similar.iloc[0]['price_inr']
    print(f" Actual Price in Dataset: ₹{actual_price:,}")
    print(f" Prediction Error: ₹{abs(predicted_price - actual_price):,.2f}")

# ============================================
# 7. BONUS: IS THE LAPTOP OVERPRICED?
# ============================================
print("\n" + "=" * 60)
print("BONUS: Price Check")
print("=" * 60)

# Check if 8GB RAM, 512GB Storage, 2.8 GHz for ₹55,000 is overpriced
bonus_specs = np.array([[8, 512, 2.8]])
fair_price = model.predict(bonus_specs)[0]
asking_price = 55000

print(f"\n Laptop Specs:")
print(f"   • RAM: 8 GB")
print(f"   • Storage: 512 GB")
print(f"   • Processor: 2.8 GHz")
print(f"\n Asking Price: ₹{asking_price:,}")
print(f" Fair Price (Predicted): ₹{fair_price:,.2f}")
print(f" Difference: ₹{asking_price - fair_price:,.2f}")

if asking_price > fair_price + 3000:
    print(f"\n OVERPRICED! This laptop is ₹{asking_price - fair_price:,.2f} above fair value.")
    print(f"   Recommendation: Negotiate or look for better deals.")
elif asking_price < fair_price - 3000:
    print(f"\n GREAT DEAL! This laptop is ₹{fair_price - asking_price:,.2f} below fair value.")
    print(f"   Recommendation: Buy it if other factors check out!")
else:
    print(f"\n FAIRLY PRICED. The price is within reasonable range.")
    print(f"   Recommendation: Good option if it meets your needs.")

# ============================================
# MODEL EQUATION
# ============================================
print("\n" + "=" * 60)
print("MODEL EQUATION")
print("=" * 60)
print(f"\nPrice = {model.intercept_:.2f}")
for i, feature in enumerate(['RAM', 'Storage', 'Processor']):
    print(f"      + {model.coef_[i]:.2f} × {feature}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE! 🎉")
print("=" * 60)