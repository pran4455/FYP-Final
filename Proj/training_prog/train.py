import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2

def handle_outliers(X):
    percentiles = np.nanpercentile(X, [1, 99], axis=0)
    return np.clip(X, percentiles[0], percentiles[1])

def load_and_preprocess_data(file_path="stress_features_all.csv"):
    df = pd.read_csv(file_path)
    X = df.drop('label', axis=1)
    y = df['label']
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, le

def create_ensemble_model():
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    svm = SVC(
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE
    )
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svm', svm)
        ],
        voting='soft'
    )
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_handler', FunctionTransformer(handle_outliers, validate=False)),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', ensemble)
    ])
    return pipeline

def plot_feature_importance(pipeline, feature_names, output_path='feature_importance.png'):
    voting_classifier = pipeline.named_steps['classifier']
    rf_model = voting_classifier.named_estimators_['rf']
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()
    
    # ==================== 5-FOLD CROSS-VALIDATION ====================
    print("\nPerforming 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracies = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\nTraining Fold {fold}/5...")
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]
        
        pipeline = create_ensemble_model()
        pipeline.fit(X_fold_train, y_fold_train)
        y_fold_pred = pipeline.predict(X_fold_val)
        
        fold_acc = accuracy_score(y_fold_val, y_fold_pred)
        fold_f1 = f1_score(y_fold_val, y_fold_pred, average='weighted')
        cv_accuracies.append(fold_acc)
        cv_f1_scores.append(fold_f1)
        
        print(f"Fold {fold} - Accuracy: {fold_acc:.4f}, F1-Score: {fold_f1:.4f}")
    
    print(f"\n5-Fold CV Results:")
    print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})")
    print(f"Mean F1-Score: {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores):.4f})")
    # =================================================================

    print("\nTraining final ensemble model on full training set...")
    pipeline = create_ensemble_model()
    pipeline.fit(X_train, y_train)
    
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)
    
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open('classification_report.txt', 'w') as f:
        f.write(f"5-Fold Cross-Validation Results:\n")
        f.write(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})\n")
        f.write(f"Mean F1-Score: {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores):.4f})\n\n")
        f.write(f"Test Set Results:\n")
        f.write(report)
    
    print("Plotting feature importance...")
    plot_feature_importance(
        pipeline,
        X_train.columns,
        'feature_importance.png'
    )
    
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        'confusion_matrix.png'
    )
    
    print("Saving model...")
    joblib.dump(pipeline, 'stress_prediction_model.joblib')
    
    print("\nTraining completed successfully!")
    print("Model saved as 'stress_prediction_model.joblib'")
    print("Visualizations saved as 'feature_importance.png' and 'confusion_matrix.png'")
    print("Classification report saved as 'classification_report.txt'")

if __name__ == "__main__":
    main()