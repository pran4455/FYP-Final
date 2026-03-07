import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

def handle_outliers(X):
    """
    Handle outliers and extreme values in the data
    """
    # Clip values to 99th percentile
    percentiles = np.nanpercentile(X, [1, 99], axis=0)
    return np.clip(X, percentiles[0], percentiles[1])

def load_and_preprocess_data(file_path="stress_features_all.csv"):
    """
    Load and preprocess the feature data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, le

def create_ensemble_model():
    """
    Create an ensemble model using Random Forest, Gradient Boosting, and SVM
    with proper preprocessing pipeline including outlier handling
    """
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    
    # SVM
    svm = SVC(
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE
    )
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svm', svm)
        ],
        voting='soft'
    )
    
    # Create pipeline with preprocessing steps
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_handler', FunctionTransformer(handle_outliers, validate=False)),
        ('scaler', RobustScaler()),  # Using RobustScaler instead of StandardScaler
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', ensemble)
    ])
    
    return pipeline

def plot_feature_importance(pipeline, feature_names, output_path='feature_importance.png'):
    """
    Plot feature importance from the Random Forest model within the pipeline
    """
    # Get the voting classifier from the pipeline
    voting_classifier = pipeline.named_steps['classifier']
    # Get the Random Forest classifier from the voting classifier
    rf_model = voting_classifier.named_estimators_['rf']
    importance = rf_model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, output_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    """
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
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()
    
    # Create and train the ensemble model
    print("Training ensemble model...")
    pipeline = create_ensemble_model()
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)
    
    # Generate and print classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    # Plot feature importance
    print("Plotting feature importance...")
    plot_feature_importance(
        pipeline,
        X_train.columns,
        'feature_importance.png'
    )
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        'confusion_matrix.png'
    )
    
    # Save the model
    print("Saving model...")
    joblib.dump(pipeline, 'stress_prediction_model.joblib')
    
    print("\nTraining completed successfully!")
    print("Model saved as 'stress_prediction_model.joblib'")
    print("Visualizations saved as 'feature_importance.png' and 'confusion_matrix.png'")
    print("Classification report saved as 'classification_report.txt'")

if __name__ == "__main__":
    main()
