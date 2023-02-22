# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle

from ml.data import process_data
from ml.model import *

import pandas as pd 

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go():
    print("================================================")
    logging.info("Loading data!")
    data = pd.read_csv('../cleaned_census.csv')

    logging.info("Splitting data")
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    logging.info("Preprocessing data!")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    logging.info("Training and hyperparameter tunning")
    model = train_model(X_train, y_train)
    pickle.dump(model, open("models/rf_model.pkl", 'wb'))
    logging.info("Inference")
    preds = inference(model, X_test)
    logging.info("Overall Metrics")
    metrics = compute_model_metrics(y_test, preds)
    logging.info(metrics)

    logging.info("Calculating metrics for sliced data")
    for category in cat_features:
        feature_values = test[category].unique()
        for value in feature_values:
            subset = test.loc[test[category] == value]
            subset_processed, y_slice, _, _  = process_data(subset, categorical_features=cat_features, label='salary', encoder=encoder, lb=lb, training=False)
            preds = inference(model, subset_processed)
            # compute model predictions for subset
            logging.info(f"Calculating model metrics, considering category {category} = {value}")
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            # calculate model metrics for subset predictions
            # write feature and metrics to file 
            slice_metrics = {}
            slice_metrics= {
                'precision': precision,
                'recall': recall,
                'f1-score': fbeta
            }
            # logging.info(slice_metrics)
            with open('slice_output.txt', 'a') as f:
                f.write(f"{category}: {value}\n")
                f.write(f"Precision: {slice_metrics['precision']:.3f}\n")
                f.write(f"Recall: {slice_metrics['recall']:.3f}\n")
                f.write(f"F1-score: {slice_metrics['f1-score']:.3f}\n\n")
                f.write("\n")