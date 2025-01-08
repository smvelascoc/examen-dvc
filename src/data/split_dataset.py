import pandas as pd
from sklearn.model_selection import train_test_split #type:ignore

def main():
    df = pd.read_csv("data/raw_data/raw.csv")
    X = df.drop(columns=["silica_concentrate", "date"])
    y = df["silica_concentrate"]
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=2406)

    pd.DataFrame(X_train).to_csv("data/processed_data/X_train.csv")
    pd.DataFrame(X_test).to_csv("data/processed_data/X_test.csv")
    pd.DataFrame(y_train).to_csv("data/processed_data/y_train.csv")
    pd.DataFrame(y_test).to_csv("data/processed_data/y_test.csv")

if __name__ == "__main__":
    main()