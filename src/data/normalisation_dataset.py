from sklearn.preprocessing import StandardScaler #type:ignore
import pandas as pd

def main():
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled).to_csv("data/processed_data/X_train_scaled.csv")
    pd.DataFrame(X_test_scaled).to_csv("data/processed_data/X_test_scaled.csv")

if __name__ == "__main__":
    main()