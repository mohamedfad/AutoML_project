import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

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

import io
import pickle



def get_models(problem_type):
    if problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

    elif problem_type == "Classification":  
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

    else:
        raise ValueError(
            "problem_type must be 'Regression' or 'Classification'."
        )

    return models



def train_and_evaluate(model,X_train,X_test,y_train,y_test,problem_type):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if problem_type == "Regression":
        mse = mean_squared_error(y_test, y_pred)

        results = {

            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mse ** 0.5,
        }

    elif problem_type == "Classification":
        results = {
            "Accuracy": accuracy_score(y_test, y_pred),

            "F1-score": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
                        "Precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
                        "Recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }

    else:
        raise ValueError(
            "problem_type must be 'Regression' or 'Classification'."
        )

    return model, results




def compare_models(selected_models,available_models,X_train,X_test,y_train,y_test,problem_type):
    results = []
    trained_models = {}

    for model_name in selected_models:
        model = available_models[model_name]

        trained_model, metrics = train_and_evaluate(
            model,
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


def select_best_model(results_df, trained_models, problem_type):
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




def save_best_model(model, preprocessor):
    model_package = {
        "model": model,
        "preprocessor": preprocessor,
    }

    buffer = io.BytesIO()
    pickle.dump(model_package, buffer)
    buffer.seek(0)

    return buffer.getvalue()