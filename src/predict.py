import pandas as pd
from data_preprocessing import preprocess_data
from model_utils import load_model

# Load test data
test_data = pd.read_csv('../data/test.csv')
X_test, _ = preprocess_data(test_data)

# Load trained model
model = load_model('../models/model.pkl')

# Predict
preds = model.predict(X_test)
out_df = test_data.copy()
out_df['predicted_yeild'] = preds

out_df.to_csv('../output/predictions.csv', index=False)
print('Predictions saved to ../output/predictions.csv')
