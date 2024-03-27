import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # For saving the model, compatible with BentoML

def load_and_prepare_data(filepath):
    ratings = pd.read_csv(filepath)
    # Simplifying for linear regression: Using user and movie IDs directly as features
    features = ratings[['userId', 'movieId']]
    target = ratings['rating']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    filepath = 'ml-latest-small/ratings.csv'  # Update with your actual path
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)
    
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Save the model for later use with BentoML
    save_model_filename = 'linear_regression_model.joblib'
    save_model(model, save_model_filename)
    print(f"Model saved to {save_model_filename}")

if __name__ == "__main__":
    main()
