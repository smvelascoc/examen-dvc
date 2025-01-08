from sklearn.linear_model import ElasticNet
import pandas as pd
import pickle

def main():
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    with open("models/best_parameters.pkl", "rb") as file:
        best_params = pickle.load(file)

    estimator = ElasticNet(**best_params)

    print("Training model")
    estimator.fit(X_train_scaled, y_train)

    with open("models/best_model.pkl", "wb") as file:
        pickle.dump(estimator, file)
    print("Best model saved at: models/best_model.pkl")

if __name__ == "__main__":
    main()