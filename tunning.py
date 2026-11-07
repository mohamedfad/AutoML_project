from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import GridSearchCV

import pandas as pd
import io
import pickle



def get_tuning_models(problem_type):

    if problem_type == "Regression":

        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

    elif problem_type == "Classification":

        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

    else:
        raise ValueError(
            "problem_type must be 'Regression' or 'Classification'."
        )

    return models



def hyperparameter(problem_type):

    if problem_type == "Regression":

        param_grids = {

            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt', 'log2']
            },

            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        }

    elif problem_type == "Classification":

        param_grids = {

            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt', 'log2']
            },

            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        }

    else:
        raise ValueError(
            "problem_type must be 'Regression' or 'Classification'."
        )

    return param_grids



def train_and_evaluate_tunning(
    model,
    param_grid,
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type
):

    # Select scoring metric
    if problem_type == "Regression":
        scoring = "neg_root_mean_squared_error"

    elif problem_type == "Classification":
        scoring = "f1_weighted"

    else:
        raise ValueError(
            "problem_type must be 'Regression' or 'Classification'."
        )

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train.values.ravel()
                    if isinstance(y_train, pd.DataFrame)
                    else y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    if problem_type == "Regression":

        mse = mean_squared_error(y_test, y_pred)

        results = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mse ** 0.5,
        }

    else:

        results = {
            "Accuracy": accuracy_score(y_test, y_pred),


            "F1-score": f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            ),
                        "Precision": precision_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            ),
        }

    return best_model, results


def compare_tunning_models(
    selected_models,
    available_models,
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type
):

    results = []
    trained_models = {}

    models = get_tuning_models(problem_type)

    for model_name in selected_models:

        # Get model
        model = models[model_name]

        # Get its parameter grid
        param_grid = available_models[model_name]

        # Tune and evaluate
        trained_model, metrics = train_and_evaluate_tunning(
            model,
            param_grid,
            X_train,
            X_test,
            y_train,
            y_test,
            problem_type
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        trained_models[model_name] = trained_model

    results_df = pd.DataFrame(results)

    return results_df, trained_models


def select_best_model_tunning(
    results_df,
    trained_models,
    problem_type
):

    if results_df.empty:
        raise ValueError("No model results are available.")

    if problem_type == "Regression":

        best_index = results_df["RMSE"].idxmin()

    elif problem_type == "Classification":

        best_index = results_df["F1-score"].idxmax()

    else:
        raise ValueError(
            "problem_type must be 'Regression' or 'Classification'."
        )

    best_model_name = results_df.loc[best_index, "Model"]

    best_model = trained_models[best_model_name]

    return best_model_name, best_model


def save_best_model_tunning(model, preprocessor):

    model_package = {
        "model": model,
        "preprocessor": preprocessor,
    }

    buffer = io.BytesIO()

    pickle.dump(model_package, buffer)

    buffer.seek(0)

    return buffer.getvalue()