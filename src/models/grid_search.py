from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
import pickle

def main():
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    estimator = ElasticNet()
    param_grid = {"alpha": np.linspace(0.05, 1, 20),
                  "l1_ratio": np.linspace(0.05,1,20)}
    
    grid = GridSearchCV(estimator=estimator,
                        param_grid=param_grid,
                        cv=5,
                        verbose=1)

    grid.fit(X_train_scaled, y_train)

    print("Best parameters")
    print(grid.best_params_)
    print("Best R2")
    print(f"{grid.best_score_:.3f}")

    with open("models/best_parameters.pkl", "wb") as file:
        pickle.dump(grid.best_params_, file)
    print("Best parameters saved at: models/best_parameters.pkl")


if __name__ == "__main__":
    main()