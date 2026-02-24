import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)


def load_housing_data():
    housing = fetch_california_housing(as_frame=True)
    features = housing.data
    target = housing.target
    
    return features, target, housing.feature_names


def perform_regression_cv(features, target):
    print("\ntask 5: cross-validation for regression")
    print("=" * 70)
    
    linear_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    forest_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=20, n_jobs=-1, max_depth=10))
    ])
    
    models = {
        'linear regression': linear_pipeline,
        'random forest': forest_pipeline
    }
    
    results_table = []
    
    for model_name, pipeline in models.items():
        print(f"\nevaluating {model_name}...")
        
        cv_scores_neg = cross_val_score(
            pipeline, features, target,
            cv=5,
            scoring='neg_root_mean_squared_error'
        )
        
        cv_scores_rmse = -cv_scores_neg
        
        cv_scores_r2 = cross_val_score(
            pipeline, features, target,
            cv=5,
            scoring='r2'
        )
        
        model_metrics = {
            'model': model_name,
            'mean_rmse': cv_scores_rmse.mean(),
            'std_dev': cv_scores_rmse.std(),
            'r2': cv_scores_r2.mean()
        }
        
        results_table.append(model_metrics)
        
        print(f"  mean rmse: {model_metrics['mean_rmse']:.4f}")
        print(f"  std dev:   {model_metrics['std_dev']:.4f}")
        print(f"  r² score:  {model_metrics['r2']:.4f}")
    
    results_dataframe = pd.DataFrame(results_table)
    print("\ncross-validation results:")
    print(results_dataframe.to_string(index=False))
    
    most_stable_idx = results_dataframe['std_dev'].idxmin()
    most_stable = results_dataframe.loc[most_stable_idx, 'model']
    print(f"\nmost stable model: {most_stable} (lowest std_dev: {results_dataframe.loc[most_stable_idx, 'std_dev']:.4f})")
    
    return results_dataframe, models


def analyze_bias_variance(features, target, models):
    print("\ntask 6: bias-variance insight")
    print("=" * 70)
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    for model_name, pipeline in models.items():
        print(f"\n{model_name}:")
        
        pipeline.fit(features_train, target_train)
        
        train_predictions = pipeline.predict(features_train)
        test_predictions = pipeline.predict(features_test)
        
        train_r2 = r2_score(target_train, train_predictions)
        test_r2 = r2_score(target_test, test_predictions)
        
        train_rmse = np.sqrt(mean_squared_error(target_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(target_test, test_predictions))
        
        print(f"  train r²:  {train_r2:.4f}")
        print(f"  test r²:   {test_r2:.4f}")
        print(f"  train rmse: {train_rmse:.4f}")
        print(f"  test rmse:  {test_rmse:.4f}")
        
        r2_gap = train_r2 - test_r2
        print(f"  r² gap:    {r2_gap:.4f}")
        
        if r2_gap > 0.1:
            print(f"  assessment: potential overfitting (train r² >> test r²)")
        else:
            print(f"  assessment: good generalization")
    
    print("\nstability comparison:")
    print("  linear regression: more stable, consistent performance")
    print("  random forest: may overfit if train r² >> test r²")


def experiment_wrong_comparison(features, target):
    print("\ntask 7: wrong comparison experiment")
    print("=" * 70)
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    pipeline_a = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=20, n_jobs=-1, max_depth=10))
    ])
    
    pipeline_b = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=20, n_jobs=-1, max_depth=10))
    ])
    
    print("\nmodel a (with random_state=42):")
    pipeline_a.fit(features_train, target_train)
    predictions_a = pipeline_a.predict(features_test)
    rmse_a = np.sqrt(mean_squared_error(target_test, predictions_a))
    r2_a = r2_score(target_test, predictions_a)
    print(f"  rmse: {rmse_a:.4f}")
    print(f"  r²:   {r2_a:.4f}")
    
    print("\nmodel b (without random_state, run 1):")
    pipeline_b.fit(features_train, target_train)
    predictions_b1 = pipeline_b.predict(features_test)
    rmse_b1 = np.sqrt(mean_squared_error(target_test, predictions_b1))
    r2_b1 = r2_score(target_test, predictions_b1)
    print(f"  rmse: {rmse_b1:.4f}")
    print(f"  r²:   {r2_b1:.4f}")
    
    print("\nmodel b (without random_state, run 2):")
    pipeline_b.fit(features_train, target_train)
    predictions_b2 = pipeline_b.predict(features_test)
    rmse_b2 = np.sqrt(mean_squared_error(target_test, predictions_b2))
    r2_b2 = r2_score(target_test, predictions_b2)
    print(f"  rmse: {rmse_b2:.4f}")
    print(f"  r²:   {r2_b2:.4f}")
    
    print("\nconclusion:")
    print("  cannot claim model a is better")
    print("  reason: model b results vary across runs")
    print("  fair comparison requires identical random seeds")
    print("  without reproducibility, differences may be random noise")


def statistical_comparison(features, target, models):
    print("\ntask 8: statistical thinking")
    print("=" * 70)
    
    linear_pipeline = models['linear regression']
    forest_pipeline = models['random forest']
    
    linear_scores = cross_val_score(
        linear_pipeline, features, target,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    linear_rmse = -linear_scores
    
    forest_scores = cross_val_score(
        forest_pipeline, features, target,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    forest_rmse = -forest_scores
    
    print("\nlinear regression cv scores (rmse):")
    for fold_idx, score in enumerate(linear_rmse, 1):
        print(f"  fold {fold_idx}: {score:.4f}")
    print(f"  mean: {linear_rmse.mean():.4f}")
    print(f"  std:  {linear_rmse.std():.4f}")
    
    print("\nrandom forest cv scores (rmse):")
    for fold_idx, score in enumerate(forest_rmse, 1):
        print(f"  fold {fold_idx}: {score:.4f}")
    print(f"  mean: {forest_rmse.mean():.4f}")
    print(f"  std:  {forest_rmse.std():.4f}")
    
    difference_per_fold = linear_rmse - forest_rmse
    
    print("\ndifference per fold (linear - forest):")
    for fold_idx, diff in enumerate(difference_per_fold, 1):
        print(f"  fold {fold_idx}: {diff:+.4f}")
    
    print(f"\nmean difference: {difference_per_fold.mean():+.4f}")
    print(f"std of difference: {difference_per_fold.std():.4f}")
    
    positive_differences = (difference_per_fold > 0).sum()
    consistent = positive_differences == 5 or positive_differences == 0
    
    print("\nstatistical assessment:")
    if consistent:
        print(f"  improvement is consistent across all folds")
        print(f"  difference appears statistically meaningful")
    else:
        print(f"  improvement varies across folds")
        print(f"  {positive_differences}/5 folds favor linear regression")
        if abs(difference_per_fold.mean()) < difference_per_fold.std():
            print(f"  mean difference < std dev")
            print(f"  difference may be random noise")
        else:
            print(f"  mean difference > std dev")
            print(f"  difference appears meaningful")


def visualize_predictions(features, target, models):
    print("\nadditional: prediction visualization")
    print("=" * 70)
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (model_name, pipeline) in enumerate(models.items()):
        pipeline.fit(features_train, target_train)
        predictions = pipeline.predict(features_test)
        
        axes[idx].scatter(target_test, predictions, alpha=0.5, s=20)
        axes[idx].plot([target_test.min(), target_test.max()], 
                      [target_test.min(), target_test.max()], 
                      'r--', lw=2)
        axes[idx].set_xlabel('actual values')
        axes[idx].set_ylabel('predicted values')
        axes[idx].set_title(f'{model_name}')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_predictions.png', dpi=300)
    print("\nprediction plots saved to: regression_predictions.png")


def main():
    print("\nlab 5: regression evaluation")
    print("=" * 70)
    
    features, target, feature_names = load_housing_data()
    print(f"\ndataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"features: {', '.join(feature_names)}")
    
    results_dataframe, models = perform_regression_cv(features, target)
    
    analyze_bias_variance(features, target, models)
    
    experiment_wrong_comparison(features, target)
    
    statistical_comparison(features, target, models)
    
    visualize_predictions(features, target, models)
    
    print("\n" + "=" * 70)
    print("regression analysis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
