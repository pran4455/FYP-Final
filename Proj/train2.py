import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional imports
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2


def handle_outliers(X):
    """
    Clip features to 1st/99th percentiles to reduce extreme outliers.
    """
    percentiles = np.nanpercentile(X, [1, 99], axis=0)
    return np.clip(X, percentiles[0], percentiles[1])


def load_and_preprocess_data(file_path="stress_features_all.csv"):
    df = pd.read_csv(file_path)
    if 'label' not in df.columns:
        raise ValueError("Expected 'label' column in the CSV")

    X = df.drop('label', axis=1)
    y = df['label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test, le


def diagnose_features(X, title='dataset'):
    """Run quick diagnostics on features: missing, constant, low-variance."""
    print(f"Dataset diagnostics for {title}:")
    print(f" - shape: {X.shape}")
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    low_variance = X.var().sort_values().head(10)
    print(f" - constant columns (<=1 unique): {constant_cols}")
    print(f" - top 10 lowest-variance columns:\n{low_variance}")
    na_counts = X.isna().sum()
    print(f" - columns with missing values: {na_counts[na_counts>0].to_dict()}")
    return constant_cols


def create_stacking_pipeline():
    """Create an imbalanced-aware pipeline with stacking ensemble.

    Base learners: LightGBM (required), CatBoost (optional), SVM fallback.
    Meta-learner: LogisticRegression.
    """
    if not HAS_LGB:
        raise ImportError("lightgbm is required for this pipeline. Install via `pip install lightgbm`.")

    estimators = []
    # Safer LightGBM defaults to avoid "no further splits" warnings
    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        min_split_gain=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    estimators.append(('lgb', lgb))

    if HAS_CAT:
        cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, verbose=0, random_seed=RANDOM_STATE)
        estimators.append(('cat', cat))
    else:
        # provide a simple fallback (SVC is slower) — we include a small RandomForest if CatBoost unavailable
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        estimators.append(('rf', rf))

    # Final meta-learner
    final_estimator = LogisticRegression(max_iter=2000, class_weight='balanced')

    stacking = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=-1, passthrough=False)

    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_handler', FunctionTransformer(handle_outliers, validate=False)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', stacking)
    ])

    return pipeline


def plot_confusion_matrix(y_true, y_pred, labels, output_path='confusion_matrix_v2.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(pipeline, feature_names, output_path='feature_importance_v2.png'):
    """Extract feature importances from a tree-based base learner if available (e.g., lightgbm).
    For stacking, we extract from the LGB estimator if present.
    """
    try:
        stacking = pipeline.named_steps['clf']
        # named_estimators_ exists after fitting
        if hasattr(stacking, 'named_estimators_'):
            named = stacking.named_estimators_
            if 'lgb' in named:
                model = named['lgb']
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    indices = np.argsort(imp)[::-1]
                    plt.figure(figsize=(12, 8))
                    plt.title('Feature Importances (LGB)')
                    plt.bar(range(len(imp)), imp[indices])
                    plt.xticks(range(len(imp)), [feature_names[i] for i in indices], rotation=90)
                    plt.tight_layout()
                    plt.savefig(output_path)
                    plt.close()
                    return
        print('No LightGBM feature importances found in stacking estimator.')
    except Exception as e:
        print('Could not extract feature importances:', e)


def main():
    print('Loading data...')
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data()

    # Quick diagnostics and feature cleaning
    consts = diagnose_features(X_train, title='X_train')
    if len(consts) > 0:
        print(f"Dropping {len(consts)} constant columns from training and test sets.")
        X_train = X_train.drop(columns=consts)
        X_test = X_test.drop(columns=[c for c in consts if c in X_test.columns])

    print('Creating pipeline...')
    pipeline = create_stacking_pipeline()

    print('Cross-validating (stratified 5-fold)')
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print('CV F1-weighted: %.4f ± %.4f' % (scores.mean(), scores.std()))
    except Exception as e:
        print('CV failed:', e)

    print('Fitting pipeline...')
    pipeline.fit(X_train, y_train)

    print('Predicting on test set...')
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print('\nClassification Report:')
    print(report)

    with open('classification_report_v2.txt', 'w') as f:
        f.write(report)

    print('Plotting confusion matrix...')
    plot_confusion_matrix(y_test, y_pred, le.classes_, 'confusion_matrix_v2.png')

    print('Plotting feature importances (if available)...')
    try:
        plot_feature_importance(pipeline, X_train.columns, 'feature_importance_v2.png')
    except Exception as e:
        print('Feature importance plotting failed:', e)

    print('Saving model...')
    joblib.dump(pipeline, 'stress_prediction_model_v2.joblib')

    print('\nDone. Artifacts: stress_prediction_model_v2.joblib, classification_report_v2.txt, confusion_matrix_v2.png (and feature_importance_v2.png if available).')


if __name__ == '__main__':
    main()
