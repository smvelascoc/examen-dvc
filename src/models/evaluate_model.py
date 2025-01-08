from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import pickle
import json

def main():
    X_test_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_train.csv")

    print("Loading model")
    estimator: ElasticNet
    with open("models/best_model.pkl", "rb") as file:
        estimator = pickle.load(file)

    y_pred = estimator.predict(X_test_scaled)

    metrics = {"R2": r2_score(y_test, y_pred),
               "MSE": mean_squared_error(y_test, y_pred)}
    
    with open("metrics/metrics.json", "w") as file:
        json.dump(metrics, file)

    pd.DataFrame(y_pred).to_csv("data/predictions.csv")

if __name__ == "__main__":
    main()