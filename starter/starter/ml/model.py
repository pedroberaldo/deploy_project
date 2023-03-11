from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from starter.ml.data import process_data

import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Use RandomizedSearchCV to find the best hyperparameters
    logging.info('Using RandomizedSearchCV to serch the best hyperparameters')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, random_state=42)
    rf_random.fit(X_train, y_train)
    
    # Print the best hyperparameters found
    print("Best Hyperparameters:", rf_random.best_params_)
    
    # Train the random forest with the best hyperparameters
    rf = RandomForestClassifier(**rf_random.best_params_, random_state=42)
    logging.info('Training the random forest with the best hyperparameters found!')
    rf.fit(X_train, y_train)
    
    return rf


def get_training_inference_pipeline():
    with open("starter/models/rf_model.pkl", 'rb') as file:
        model = pickle.load(file)
    pipe = Pipeline(
            steps=[
                ("preprocessor", process_data),
                ("classifier", model)
            ]
        )
    return pipe

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(X): ##### CREATE A INFERECE PIPELINE #####
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pipeline = get_training_inference_pipeline()
    preds = pipeline.predict(X) 
    return preds
    
def xgboost(X_train, X_test, y_train, y_test):
    params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [50, 100, 200]
    }

    # Define a function for training an XGBoost model with a given set of hyperparameters
    def train_xgboost(params, X_train, y_train):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

    # Perform a grid search over the hyperparameters
    clf = GridSearchCV(xgb.XGBClassifier(), params, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)

    # Print the best hyperparameters and corresponding accuracy
    print("Best hyperparameters: ", clf.best_params_)
    print("Accuracy: ", clf.best_score_)

    # Train a final model with the best hyperparameters
    params = clf.best_params_
    model = train_xgboost(params, X_train, y_train)

    # Predict the validation set and compute accuracy
    preds = model.predict(X_test)
    metrics = compute_model_metrics(y_test, preds)
    with open('slice_output.txt', 'w') as f:
        # f.write(f"{category}: {value}\n")
        f.write(f"Precision: {metrics[0]:.3f}\n")
        f.write(f"Recall: {metrics[1]:.3f}\n")
        f.write(f"F1-score: {metrics[2]:.3f}\n\n")


    print(f"Precision: {metrics[0]}")
    print(f"Recall: {metrics[1]}")
    print(f"F1-score: {metrics[2]}")
