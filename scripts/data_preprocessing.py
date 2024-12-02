import pandas as pd

def load_and_preprocess_data():
    # Load data
    data_red = pd.read_csv('../data/winequality-red.csv', delimiter=';')
    data_white = pd.read_csv('../data/winequality-white.csv', delimiter=';')

    # Add wine type
    data_red['wine_type'] = 'red'
    data_white['wine_type'] = 'white'

    # Combine datasets
    data = pd.concat([data_red, data_white], axis=0)

    # One-hot encode wine type
    data = pd.get_dummies(data, columns=['wine_type'], drop_first=True)

    # Drop missing values
    data = data.dropna()

    return data

if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("Data successfully loaded and preprocessed.")

