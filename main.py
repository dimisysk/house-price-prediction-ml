# main.py

import pandas as pd

# imports από utils
from house_price_prediction_ensemble import (
    load_data,
    split_data,
    run_initial_eda,
    plot_target_distribution,
    print_target_insight,
    create_preprocessor,
    evaluate_dummy,
    evaluate_linear_regression,
    evaluate_decision_tree,
    evaluate_decision_tree_depth,
    plot_tree_performance,
    evaluate_random_forest,
    plot_rf_performance,
    evaluate_gradient_boosting,
    plot_gb_results,
    evaluate_xgboost,
    plot_xgb_results,
    plot_feature_importance,
    plot_residuals,
    create_submission
)


def main():
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df, test_df = load_data(train_path, test_path)
    X, y, test_features, test_ids = split_data(train_df, test_df)

    run_initial_eda(train_df, test_df, X, y)
    plot_target_distribution(y)
    print_target_insight(y)

    preprocessor = create_preprocessor(X)

    dummy_results = evaluate_dummy(X, y, preprocessor)
    lr_results = evaluate_linear_regression(X, y, preprocessor)
    dt_results = evaluate_decision_tree(X, y, preprocessor, max_depth=6, min_samples_leaf=5)

    print("\n=== Baseline Results (RMSLE) ===")
    print(
        f"Dummy Regressor: {dummy_results['rmse_mean']:.4f} "
        f"(+/- {dummy_results['rmse_std']:.4f}) | "
        f"Runtime: {dummy_results['runtime_sec']:.2f} sec"
    )

    print(
        f"Linear Regression: {lr_results['rmse_mean']:.4f} "
        f"(+/- {lr_results['rmse_std']:.4f}) | "
        f"Runtime: {lr_results['runtime_sec']:.2f} sec"
    )

    print(
        f"Decision Tree: {dt_results['rmse_mean']:.4f} "
        f"(+/- {dt_results['rmse_std']:.4f}) | "
        f"Runtime: {dt_results['runtime_sec']:.2f} sec"
    )

    depths = list(range(1, 21))

    train_scores, cv_scores = evaluate_decision_tree_depth(
        X, y, preprocessor, depths
    )

    plot_tree_performance(depths, train_scores, cv_scores)

    n_estimators_list = [50, 100, 200, 300]

    rf_results = evaluate_random_forest(
        X, y, preprocessor, n_estimators_list
    )

    print("\n=== Random Forest Results ===")
    print(rf_results)

    plot_rf_performance(rf_results)

    gb_param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 600]
    }

    gb_results = evaluate_gradient_boosting(
        X, y, preprocessor, gb_param_grid
    )

    print("\n=== Gradient Boosting Results ===")
    print(gb_results)

    plot_gb_results(gb_results)

    xgb_param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 600]
    }

    xgb_results = evaluate_xgboost(
        X, y, preprocessor, xgb_param_grid
    )

    print("\n=== XGBoost Results ===")
    print(xgb_results)

    plot_xgb_results(xgb_results)

    print("\n=== Top 10 Permutation Importances ===")
    top10_importances = plot_feature_importance(X, y, preprocessor)
    print(top10_importances)

    plot_residuals(X, y, preprocessor)

    submission = create_submission(X, y, test_features, test_ids, preprocessor)
    submission.to_csv("submission.csv", index=False, float_format="%.2f")

    print("\nSubmission file created!")

    print("\nSubmission Preview:")
    print(submission.head())

    print("\nSubmission Summary:")
    print(submission.describe())



    comparison_df = pd.DataFrame({
        "Model": [
            "Dummy",
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ],
        "RMSE": [
            dummy_results["rmse_mean"],
            lr_results["rmse_mean"],
            dt_results["rmse_mean"],
            rf_results.loc[rf_results["n_estimators"] == 200, "rmse_mean"].values[0],
            gb_results.loc[
                (gb_results["learning_rate"] == 0.10) & (gb_results["n_estimators"] == 300),
                "rmse_mean"
            ].values[0],
            xgb_results.loc[
                (xgb_results["learning_rate"] == 0.01) & (xgb_results["n_estimators"] == 600),
                "rmse_mean"
            ].values[0]
        ],
        "Std": [
            dummy_results["rmse_std"],
            lr_results["rmse_std"],
            dt_results["rmse_std"],
            rf_results.loc[rf_results["n_estimators"] == 200, "rmse_std"].values[0],
            gb_results.loc[
                (gb_results["learning_rate"] == 0.10) & (gb_results["n_estimators"] == 300),
                "rmse_std"
            ].values[0],
            xgb_results.loc[
                (xgb_results["learning_rate"] == 0.01) & (xgb_results["n_estimators"] == 600),
                "rmse_std"
            ].values[0]
        ],
        "Runtime_sec": [
            dummy_results["runtime_sec"],
            lr_results["runtime_sec"],
            dt_results["runtime_sec"],
            rf_results.loc[rf_results["n_estimators"] == 200, "runtime_sec"].values[0],
            gb_results.loc[
                (gb_results["learning_rate"] == 0.10) & (gb_results["n_estimators"] == 300),
                "runtime_sec"
            ].values[0],
            xgb_results.loc[
                (xgb_results["learning_rate"] == 0.01) & (xgb_results["n_estimators"] == 600),
                "runtime_sec"
            ].values[0]
        ]
    })

    comparison_df["RMSE"] = comparison_df["RMSE"].round(4)
    comparison_df["Std"] = comparison_df["Std"].round(4)
    comparison_df["Runtime_sec"] = comparison_df["Runtime_sec"].round(2)

    print(comparison_df)




if __name__ == "__main__":
    main()