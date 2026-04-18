import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
import time







# Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# Data Loading
def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


# Data Splitting (Features / Target / IDs)
def split_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    test_ids = test_df["Id"].copy()

    train_df = train_df.drop("Id", axis=1)
    test_df = test_df.drop("Id", axis=1)

    y = train_df["SalePrice"].copy()
    X = train_df.drop("SalePrice", axis=1).copy()

    return X, y, test_df, test_ids



# Dataset Shape Information
def get_dataset_shapes(train_df: pd.DataFrame, test_df: pd.DataFrame):
    return train_df.shape, test_df.shape



# Feature Type Identification (Numeric / Categorical)
def get_feature_type_counts(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return {
        "numeric_count": len(numeric_features),
        "categorical_count": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }



# Utility Function (Currency Formatting)
def format_currency(value):
    return f"${value:,.2f}"



# Target Variable Summary (SalePrice Analysis)
def get_target_summary(y: pd.Series):
    return {
        "mean": format_currency(y.mean()),
        "median": format_currency(y.median()),
        "std": format_currency(y.std()),
        "skewness": f"{y.skew():.4f}",
        "min": format_currency(y.min()),
        "max": format_currency(y.max())
    }


# Missing Values Analysis (Top Features)
def get_top_missing_columns(X: pd.DataFrame, top_n: int = 10):
    missing_counts = X.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    missing_percent = (X.isnull().mean() * 100)
    missing_percent = missing_percent[missing_percent > 0]

    missing_table = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": missing_percent[missing_counts.index]
    })

    return missing_table.head(top_n)


# Visualization (Target Distribution)
def plot_target_distribution(y: pd.Series):
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=50)
    plt.title("SalePrice Distribution")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.show()


# Initial Exploratory Data Analysis (EDA)
def run_initial_eda(train_df: pd.DataFrame, test_df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    train_shape, test_shape = get_dataset_shapes(train_df, test_df)
    feature_info = get_feature_type_counts(X)
    target_summary = get_target_summary(y)
    top_missing = get_top_missing_columns(X)

    print("Train shape:", train_shape)
    print("Test shape:", test_shape)

    print("\nTarget Summary (SalePrice):")
    for key, value in target_summary.items():
        print(f"{key}: {value}")

    print("\nFeature Type Counts:")
    print("Numeric features:", feature_info["numeric_count"])
    print("Categorical features:", feature_info["categorical_count"])

    print("\nTop 10 Columns by Missingness:")
    print(top_missing)



# Target Insight (Skewness Interpretation)
def print_target_insight(y: pd.Series):
    skew = y.skew()

    if skew > 1:
        print("\nInsight: Strong positive skew → consider log transformation (RMSLE justified)")
    elif skew > 0.5:
        print("\nInsight: Moderate skew → log transformation may help")
    else:
        print("\nInsight: Low skew → transformation likely unnecessary")


# Feature Grouping for Preprocessing
def get_feature_groups(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features




def create_preprocessor(X: pd.DataFrame):
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor



# Target Transformation (Log for RMSLE)
def transform_target(y: pd.Series):
    return np.log1p(y)




# RMSLE Scorer (via RMSE on log target)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


# Dummy Regressor Baseline
def evaluate_dummy(X, y, preprocessor):

    y_log = transform_target(y)

    model = DummyRegressor(strategy="mean")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=5,
        scoring=rmse_scorer
    )

    runtime = time.time() - start_time

    return {
        "rmse_mean": -scores.mean(),
        "rmse_std": scores.std(),
        "runtime_sec": runtime
    }




# Linear Regression Baseline
def evaluate_linear_regression(X, y, preprocessor):

    y_log = transform_target(y)

    model = LinearRegression()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=5,
        scoring=rmse_scorer
    )

    runtime = time.time() - start_time

    return {
        "rmse_mean": -scores.mean(),
        "rmse_std": scores.std(),
        "runtime_sec": runtime
    }



# RMSE (for log target)
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



# Decision Tree Depth Analysis
def evaluate_decision_tree_depth(X, y, preprocessor, depths, min_samples_leaf=5):

    y_log = transform_target(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    train_scores = []
    cv_scores = []

    for depth in depths:

        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            random_state=0
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        fold_train_rmse = []
        fold_val_rmse = []

        for train_idx, val_idx in kf.split(X):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

            pipeline.fit(X_train, y_train)

            # Train prediction
            y_train_pred = pipeline.predict(X_train)
            train_rmse = compute_rmse(y_train, y_train_pred)

            # Validation prediction
            y_val_pred = pipeline.predict(X_val)
            val_rmse = compute_rmse(y_val, y_val_pred)

            fold_train_rmse.append(train_rmse)
            fold_val_rmse.append(val_rmse)

        train_scores.append(np.mean(fold_train_rmse))
        cv_scores.append(np.mean(fold_val_rmse))

    return train_scores, cv_scores




# Plot Depth vs Performance
def plot_tree_performance(depths, train_scores, cv_scores):

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, marker='o', label="Train RMSE")
    plt.plot(depths, cv_scores, marker='o', label="CV RMSE")

    plt.xlabel("Max Depth")
    plt.ylabel("RMSE (log scale)")
    plt.title("Decision Tree: Depth vs Performance")
    plt.legend()
    plt.grid()

    plt.show()




# Random Forest Evaluation
def evaluate_random_forest(X, y, preprocessor, n_estimators_list):

    y_log = transform_target(y)

    results = []

    for n in n_estimators_list:

        model = RandomForestRegressor(
            n_estimators=n,
            random_state=0,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        start_time = time.time()

        scores = cross_val_score(
            pipeline,
            X,
            y_log,
            cv=5,
            scoring=rmse_scorer
        )

        runtime = time.time() - start_time

        results.append({
            "n_estimators": n,
            "rmse_mean": -scores.mean(),
            "rmse_std": scores.std(),
            "runtime_sec": runtime
        })

    return pd.DataFrame(results)





# Gradient Boosting Evaluation
def evaluate_gradient_boosting(X, y, preprocessor, param_grid):

    y_log = transform_target(y)

    results = []

    for lr in param_grid["learning_rate"]:
        for n in param_grid["n_estimators"]:

            model = GradientBoostingRegressor(
                learning_rate=lr,
                n_estimators=n,
                max_depth=3,
                random_state=0
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            start_time = time.time()

            scores = cross_val_score(
                pipeline,
                X,
                y_log,
                cv=5,
                scoring=rmse_scorer
            )

            runtime = time.time() - start_time

            results.append({
                "learning_rate": lr,
                "n_estimators": n,
                "rmse_mean": -scores.mean(),
                "rmse_std": scores.std(),
                "runtime_sec": runtime
            })

    return pd.DataFrame(results)




# Plot RF Performance
def plot_rf_performance(results_df):

    plt.figure(figsize=(8,5))

    plt.plot(results_df["n_estimators"], results_df["rmse_mean"], marker='o')

    plt.xlabel("Number of Trees")
    plt.ylabel("RMSE (log scale)")
    plt.title("Random Forest Performance")
    plt.grid()

    plt.show()


# Plot Learning Rate vs Performance
def plot_gb_results(results_df):

    plt.figure(figsize=(8,5))

    for lr in results_df["learning_rate"].unique():
        subset = results_df[results_df["learning_rate"] == lr]

        plt.plot(
            subset["n_estimators"],
            subset["rmse_mean"],
            marker='o',
            label=f"lr={lr}"
        )

    plt.xlabel("Number of Trees")
    plt.ylabel("RMSE (log scale)")
    plt.title("Gradient Boosting: Learning Rate Trade-off")
    plt.legend()
    plt.grid()

    plt.show()




# XGBoost Evaluation
def evaluate_xgboost(X, y, preprocessor, param_grid):

    y_log = transform_target(y)

    results = []

    for lr in param_grid["learning_rate"]:
        for n in param_grid["n_estimators"]:

            model = XGBRegressor(
                learning_rate=lr,
                n_estimators=n,
                tree_method="hist",
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=0,
                n_jobs=-1
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            start_time = time.time()

            scores = cross_val_score(
                pipeline,
                X,
                y_log,
                cv=5,
                scoring=rmse_scorer
            )

            runtime = time.time() - start_time

            results.append({
                "learning_rate": lr,
                "n_estimators": n,
                "rmse_mean": -scores.mean(),
                "rmse_std": scores.std(),
                "runtime_sec": runtime
            })

    return pd.DataFrame(results)




# Final Model Training & Submission
def create_submission(X, y, X_test, test_ids, preprocessor):

    y_log = transform_target(y)

    # best model (από GB)
    model = GradientBoostingRegressor(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        random_state=0
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # TRAIN ΣΕ ΟΛΟ ΤΟ DATASET
    pipeline.fit(X, y_log)

    # PREDICT (σε log scale)
    preds_log = pipeline.predict(X_test)

    # inverse transform (ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ)
    preds = np.expm1(preds_log)
    preds = np.round(preds, 2)

    # submission dataframe
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })

    return submission



def plot_feature_importance(X, y, preprocessor):

    y_log = transform_target(y)

    model = GradientBoostingRegressor(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        random_state=0
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y_log)

    result = permutation_importance(
        pipeline,
        X,
        y_log,
        n_repeats=5,
        random_state=0,
        n_jobs=-1
    )

    importances = result.importances_mean
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top10 = importance_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10["feature"], top10["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances (Permutation)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return top10




# Residual Analysis
def plot_residuals(X, y, preprocessor):

    y_log = transform_target(y)

    model = GradientBoostingRegressor(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        random_state=0
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y_log)

    y_pred = pipeline.predict(X)
    residuals = y_log - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted (log price)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predictions")
    plt.tight_layout()
    plt.show()




# Plot XGBoost Performance
def plot_xgb_results(results_df):

    plt.figure(figsize=(8,5))

    for lr in results_df["learning_rate"].unique():
        subset = results_df[results_df["learning_rate"] == lr]

        plt.plot(
            subset["n_estimators"],
            subset["rmse_mean"],
            marker='o',
            label=f"lr={lr}"
        )

    plt.xlabel("Number of Trees")
    plt.ylabel("RMSE (log scale)")
    plt.title("XGBoost: Learning Rate Trade-off")
    plt.legend()
    plt.grid()

    plt.show()





# Final Decision Tree Evaluation
def evaluate_decision_tree(X, y, preprocessor, max_depth=6, min_samples_leaf=5):

    y_log = transform_target(y)

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=5,
        scoring=rmse_scorer
    )

    runtime = time.time() - start_time

    return {
        "rmse_mean": -scores.mean(),
        "rmse_std": scores.std(),
        "runtime_sec": runtime
    }






if __name__ == "__main__":
    main()