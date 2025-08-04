import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import preprocess_data
from model_utils import save_model

# Load data
data = pd.read_csv('../data/train.csv')

# Preprocess
X, y = preprocess_data(data)

# Split for local validation (optional)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

# Save model
save_model(model, '../models/model.pkl')

# Validation
score = model.score(X_val, y_val)
with open('../output/training_log.txt', 'w') as f:
    f.write(f"Validation R2 Score: {score:.4f}\n")
print(f"Validation R2 Score: {score:.4f}")
