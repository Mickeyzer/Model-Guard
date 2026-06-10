import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Splitting dataset (test_size=0.2)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Training and saving complete.")

if __name__ == "__main__":
    main()
