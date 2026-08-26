import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import sqlite3
import os

os.makedirs('models', exist_ok=True)

print("Loading data...")
conn = sqlite3.connect('data/inventory.db')
df = pd.read_sql("SELECT * FROM inventory_history", conn)
conn.close()

print("Engineering features...")
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Calculate rolling averages
df['avg_7d'] = df.groupby('sku')['daily_demand'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['avg_30d'] = df.groupby('sku')['daily_demand'].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

# Target: next day's demand
df['target'] = df.groupby('sku')['daily_demand'].shift(-1)
df = df.dropna()

# Features and target
features = ['day_of_week', 'month', 'avg_7d', 'avg_30d', 'current_stock']
X = df[features]
y = df['target']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Score
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Model trained! MAPE: {mape:.2f}%")

# Save
with open('models/demand_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/feature_schema.pkl', 'wb') as f:
    pickle.dump({'features': features}, f)

print("Model saved to models/demand_model.pkl")