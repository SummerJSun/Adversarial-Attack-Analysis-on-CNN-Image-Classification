import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(X, y):
    # Load and split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    
    # Train models
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    print("Training Decision Tree...")
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    
    # Calculate accuracies
    rf_accuracy = accuracy_score(y_test, rf_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    print("\n=== Random Forest Results ===")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\n=== Decision Tree Results ===")
    print(f"Accuracy: {dt_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, dt_pred))
    
    # Plot confusion matrices
    plt.figure(figsize=(15, 5))
    
    # Random Forest confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Decision Tree confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Decision Tree Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('ml_model_results.png')
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X.shape[1])],
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Top 10 Most Important Features (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return rf_model, dt_model

if __name__ == "__main__":
    # Load the data
    X = np.load('/Users/jinjiahui/Desktop/CS470Project/embed_adversarial_cifar10/X_features.npy')
    y = np.load('/Users/jinjiahui/Desktop/CS470Project/embed_adversarial_cifar10/y_labels.npy')
    
    # Run evaluation
    rf_model, dt_model = evaluate_models(X, y)
    
