
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def split_features_target(df,target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y



def detect_problem_type(df, target_column):

    is_numeric = pd.api.types.is_numeric_dtype(df[target_column])

    unique_count = df[target_column].nunique()

    if not is_numeric:
        return "Classification"

    elif unique_count <= 10:
        return "Classification"

    else:
        return "Regression"


def build_preprocessor(X):
    """Build preprocessing from the columns in the user's feature table."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError(
            "No supported feature columns found. The dataset needs numeric "
            "or text/category columns besides the target."
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")



def split_train_test(X,y,test_size=0.2,random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
