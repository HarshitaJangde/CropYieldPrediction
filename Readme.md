# Crop Yield Prediction ML Project

## Project Overview

This project predicts crop yield based on soil and weather features such as Fertilizer, Temperature, N, P, and K content. A Random Forest regression model is used to estimate yield from the provided dataset.

---

## Directory Structure

MLProjectName/
│
├── src/ # Source code files (training, prediction, preprocessing)
├── data/ # Dataset files: original and train/test splits
├── models/ # Trained machine learning model files
├── output/ # Outputs: predictions and logs
├── docs/ # Documentation and project report
├── notebooks/ # Jupyter notebooks for data preparation and EDA
├── requirements.txt # Python dependencies


---

## Setup Instructions

1. **Install Dependencies**

Open your terminal and run:

`pip install -r requirements.txt`


2. **Prepare Data**

Place your original dataset `data.csv` in the `data/` folder.

Run the notebook `notebooks/split_data.ipynb` to create `train.csv` and `test.csv` inside the `data/` directory.

---

## How to Run

### Train the Model

`cd src`
`python train.py`


This script will:

- Load `train.csv`.
- Train a Random Forest regressor.
- Save the trained model as `models/model.pkl`.
- Write training log to `output/training_log.txt`.

---

### Generate Predictions

This script will:

- Load `test.csv`.
- Load the saved model.
- Predict yield values.
- Save predictions in `output/predictions.csv`.

---

## Outputs

- **Training Log:** `output/training_log.txt` — shows validation performance.
- **Predictions:** `output/predictions.csv` — input test data with predicted yields appended.

---

## Notes

- The notebook `split_data.ipynb` shuffles and splits your original data for better model training and evaluation.
- You can modify or extend the model in `src/train.py` and `src/model_utils.py`.
- See `docs/report.pdf` for detailed methodology and results.

---
