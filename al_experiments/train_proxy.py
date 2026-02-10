from ioh import ProblemClass, get_problem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from al_experiments.pbo import generate_random_data
import pickle
import os

def train_random_forest(X, y, n_estimators=100, test_size=0.2, random_state=42):
    """
    Train a Random Forest Regressor on the provided dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input features.
    y : array-like, shape (n_samples,)
        The target values.
    n_estimators : int, optional (default=100)
        The number of trees in the forest.
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
    random_state : int, optional (default=42)
        Controls the randomness of the estimator.

    Returns
    -------
    model : RandomForestRegressor
        The trained Random Forest model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    print("Random Forest Regressor model trained successfully.")
    print(f"\tTraining R-squared: {model.score(X_train, y_train):.4f}")
    print(f"\tTest R-squared: {model.score(X_test, y_test):.4f}")

    return model

def save_model(model, filepath, metadata=None):
    """
    Save the trained model to a file.

    Parameters
    ----------
    model : object
        The trained model.
    filepath : str
        The path to the file where the model will be saved.
    metadata : dict, optional
        Additional metadata to save with the model.
    """
    model.metadata = metadata

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}.")
    return filepath

if __name__ == "__main__":
    problem = get_problem("OneMax", 1, 20, ProblemClass.PBO)
    X, y = generate_random_data(problem, [32]*4, n_samples=100)

    
    rf_model = RandomForestRegressor(random_state=42)
    
    grid_search_params = {
        "n_estimators": [100],
        #"n_estimators": [2, 5, 10, 50, 100, 500, 1000],
        #"max_depth": [None, 10, 20],
        #"min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=grid_search_params, cv=5, n_jobs=-1, scoring='r2', verbose=3)
    grid_search.fit(X, y)
    print(f"Best parameters found: {grid_search.best_params_}")
