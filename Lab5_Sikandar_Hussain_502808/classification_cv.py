import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            RocCurveDisplay)
import joblib

random.seed(42)
np.random.seed(42)


def load_titanic_data():
    dataset = sns.load_dataset("titanic")
    dataset = dataset.drop(columns=["alive", "embark_town", "class", "who"], errors='ignore')
    
    features = dataset.drop("survived", axis=1)
    target = dataset["survived"]
    
    return features, target


def create_preprocessing_pipeline():
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_exclude=np.number))
        ])
    
    return preprocessor


def test_reproducibility(features, target):
    print("step 1: reproducibility setup")
    print("=" * 70)
    
    preprocessor = create_preprocessing_pipeline()
    model = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    print("\nexperiment 1 (with random_state=42):")
    model.fit(features_train, target_train)
    predictions_first = model.predict(features_test)
    accuracy_first = accuracy_score(target_test, predictions_first)
    print(f"  accuracy: {accuracy_first:.6f}")
    
    model.fit(features_train, target_train)
    predictions_second = model.predict(features_test)
    accuracy_second = accuracy_score(target_test, predictions_second)
    print(f"  accuracy: {accuracy_second:.6f}")
    
    identical = np.array_equal(predictions_first, predictions_second)
    print(f"\n  results identical: {identical}")
    
    print("\nexperiment 2 (without random_state):")
    model_no_seed = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    
    model_no_seed.fit(features_train, target_train)
    predictions_third = model_no_seed.predict(features_test)
    accuracy_third = accuracy_score(target_test, predictions_third)
    print(f"  run 1 accuracy: {accuracy_third:.6f}")
    
    model_no_seed.fit(features_train, target_train)
    predictions_fourth = model_no_seed.predict(features_test)
    accuracy_fourth = accuracy_score(target_test, predictions_fourth)
    print(f"  run 2 accuracy: {accuracy_fourth:.6f}")
    
    different = not np.array_equal(predictions_third, predictions_fourth)
    print(f"\n  results different: {different}")
    print()


def perform_cross_validation(features, target):
    print("\ntask 1: k-fold cross validation")
    print("=" * 70)
    
    preprocessor = create_preprocessing_pipeline()
    
    models = {
        'logistic regression': Pipeline([
            ('preprocess', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'random forest': Pipeline([
            ('preprocess', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
        ]),
        'svm': Pipeline([
            ('preprocess', preprocessor),
            ('classifier', SVC(random_state=42, probability=True))
        ])
    }
    
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results_table = []
    
    for model_name, pipeline in models.items():
        print(f"\nevaluating {model_name}...")
        
        cv_results = cross_validate(
            pipeline, features, target, 
            cv=5, 
            scoring=scoring_metrics,
            return_train_score=False
        )
        
        model_metrics = {
            'model': model_name,
            'accuracy': cv_results['test_accuracy'].mean(),
            'precision': cv_results['test_precision'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'f1': cv_results['test_f1'].mean(),
            'roc_auc': cv_results['test_roc_auc'].mean(),
            'std_dev': cv_results['test_f1'].std()
        }
        
        results_table.append(model_metrics)
        
        print(f"  accuracy:  {model_metrics['accuracy']:.4f}")
        print(f"  precision: {model_metrics['precision']:.4f}")
        print(f"  recall:    {model_metrics['recall']:.4f}")
        print(f"  f1:        {model_metrics['f1']:.4f}")
        print(f"  roc_auc:   {model_metrics['roc_auc']:.4f}")
        print(f"  std_dev:   {model_metrics['std_dev']:.4f}")
    
    results_dataframe = pd.DataFrame(results_table)
    print("\ncross validation results:")
    print(results_dataframe.to_string(index=False))
    
    most_stable_idx = results_dataframe['std_dev'].idxmin()
    most_stable = results_dataframe.loc[most_stable_idx, 'model']
    print(f"\nmost stable model: {most_stable} (lowest std_dev: {results_dataframe.loc[most_stable_idx, 'std_dev']:.4f})")
    
    return results_dataframe, models


def plot_roc_curves(features, target, models, results_dataframe):
    print("\ntask 2: roc curve visualization")
    print("=" * 70)
    
    top_models = results_dataframe.nlargest(2, 'roc_auc')
    top_model_names = top_models['model'].tolist()
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    plt.figure(figsize=(10, 6))
    
    for model_name in top_model_names:
        pipeline = models[model_name]
        pipeline.fit(features_train, target_train)
        
        display = RocCurveDisplay.from_estimator(
            pipeline, features_test, target_test,
            name=model_name,
            alpha=0.8
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='random classifier')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('roc curves for top 2 models')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    print("\nroc curves saved to: roc_curves.png")
    
    print(f"\ntop models by roc_auc:")
    for idx, row in top_models.iterrows():
        print(f"  {row['model']}: {row['roc_auc']:.4f}")


def analyze_threshold(features, target, models, best_model_name):
    print("\ntask 3: threshold analysis")
    print("=" * 70)
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    pipeline = models[best_model_name]
    pipeline.fit(features_train, target_train)
    
    probabilities = pipeline.predict_proba(features_test)[:, 1]
    
    threshold_default = 0.5
    predictions_default = (probabilities > threshold_default).astype(int)
    precision_default = precision_score(target_test, predictions_default)
    recall_default = recall_score(target_test, predictions_default)
    
    threshold_custom = 0.3
    predictions_custom = (probabilities > threshold_custom).astype(int)
    precision_custom = precision_score(target_test, predictions_custom)
    recall_custom = recall_score(target_test, predictions_custom)
    
    print(f"\nmodel: {best_model_name}")
    print(f"\nthreshold = {threshold_default}:")
    print(f"  precision: {precision_default:.4f}")
    print(f"  recall:    {recall_default:.4f}")
    
    print(f"\nthreshold = {threshold_custom}:")
    print(f"  precision: {precision_custom:.4f}")
    print(f"  recall:    {recall_custom:.4f}")
    
    print(f"\nthreshold impact:")
    print(f"  precision change: {precision_custom - precision_default:+.4f}")
    print(f"  recall change:    {recall_custom - recall_default:+.4f}")
    
    print("\nfor fraud detection:")
    print("  lower threshold (0.3) is safer")
    print("  reason: higher recall catches more fraud cases")
    print("  even though precision drops, missing fraud is costlier")


def analyze_errors(features, target, models, best_model_name):
    print("\ntask 4: error analysis")
    print("=" * 70)
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    pipeline = models[best_model_name]
    pipeline.fit(features_train, target_train)
    predictions = pipeline.predict(features_test)
    
    confusion = confusion_matrix(target_test, predictions)
    true_negative, false_positive, false_negative, true_positive = confusion.ravel()
    
    print(f"\nmodel: {best_model_name}")
    print(f"\nconfusion matrix:")
    print(f"  true negatives:  {true_negative}")
    print(f"  false positives: {false_positive}")
    print(f"  false negatives: {false_negative}")
    print(f"  true positives:  {true_positive}")
    
    print(f"\nerror breakdown:")
    print(f"  false positives: {false_positive} (predicted survived, actually died)")
    print(f"  false negatives: {false_negative} (predicted died, actually survived)")
    
    misclassified_mask = target_test != predictions
    misclassified_samples = features_test[misclassified_mask]
    
    print(f"\ntotal misclassified samples: {misclassified_mask.sum()}")
    print(f"\nexamining first 5 difficult samples:")
    print(misclassified_samples.head())
    
    total_samples = len(target_test)
    class_distribution = target_test.value_counts()
    majority_class = class_distribution.idxmax()
    majority_count = class_distribution.max()
    
    print(f"\nclass distribution in test set:")
    print(f"  class 0 (died):     {class_distribution[0]}")
    print(f"  class 1 (survived): {class_distribution[1]}")
    print(f"  majority class:     {majority_class}")
    
    if false_positive > false_negative:
        print(f"\nbias assessment:")
        print(f"  model has more false positives than false negatives")
        print(f"  suggests slight bias toward predicting survival")
    else:
        print(f"\nbias assessment:")
        print(f"  model has more false negatives than false positives")
        print(f"  suggests slight bias toward predicting death")


def save_best_model(features, target, models, best_model_name):
    print("\npart 4: serialization & reproducibility")
    print("=" * 70)
    
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    best_pipeline = models[best_model_name]
    best_pipeline.fit(features_train, target_train)
    
    joblib.dump(best_pipeline, 'best_classifier.joblib')
    print(f"\nbest model ({best_model_name}) saved to: best_classifier.joblib")
    
    loaded_pipeline = joblib.load('best_classifier.joblib')
    original_predictions = best_pipeline.predict(features_test)
    loaded_predictions = loaded_pipeline.predict(features_test)
    
    predictions_match = np.array_equal(original_predictions, loaded_predictions)
    print(f"\nverification:")
    print(f"  loaded model predictions match: {predictions_match}")
    
    original_accuracy = accuracy_score(target_test, original_predictions)
    loaded_accuracy = accuracy_score(target_test, loaded_predictions)
    
    print(f"  original accuracy: {original_accuracy:.4f}")
    print(f"  loaded accuracy:   {loaded_accuracy:.4f}")


def main():
    print("\nlab 5: model evaluation and reproducibility")
    print("=" * 70)
    
    features, target = load_titanic_data()
    print(f"\ndataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
    
    test_reproducibility(features, target)
    
    results_dataframe, models = perform_cross_validation(features, target)
    
    best_model_idx = results_dataframe['roc_auc'].idxmax()
    best_model_name = results_dataframe.loc[best_model_idx, 'model']
    
    plot_roc_curves(features, target, models, results_dataframe)
    
    analyze_threshold(features, target, models, best_model_name)
    
    analyze_errors(features, target, models, best_model_name)
    
    save_best_model(features, target, models, best_model_name)
    
    print("\n" + "=" * 70)
    print("classification analysis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
