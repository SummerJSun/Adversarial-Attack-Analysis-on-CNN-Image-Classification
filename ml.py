import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_adversarial_impact(pre_adv_path, post_adv_path, random_state=42):
    """
    Evaluate model performance on both pre and post adversarial samples
    using the same train/test split for fair comparison.
    """
    # Load pre-adversarial data
    X_pre = np.load(f'{pre_adv_path}/X_features.npy')
    y_pre = np.load(f'{pre_adv_path}/y_labels.npy')
    
    # Load post-adversarial data
    X_post = np.load(f'{post_adv_path}/X_features.npy')
    y_post = np.load(f'{post_adv_path}/y_labels.npy')
    
    # Create train/test indices - use the same split for both datasets
    indices = np.arange(len(X_pre))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=random_state,
        stratify=y_pre  # Ensure balanced classes
    )
    
    # Split pre-adversarial data
    X_train = X_pre[train_indices]
    y_train = y_pre[train_indices]
    X_test_pre = X_pre[test_indices]
    y_test_pre = y_pre[test_indices]
    
    # Get corresponding post-adversarial test samples
    X_test_post = X_post[test_indices]
    y_test_post = y_post[test_indices]
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state)
    }
    
    results = {}

    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on pre-adversarial test set
        pre_pred = model.predict(X_test_pre)
        pre_acc = accuracy_score(y_test_pre, pre_pred)
        
        # Evaluate on post-adversarial test set
        post_pred = model.predict(X_test_post)
        post_acc = accuracy_score(y_test_post, post_pred)
        
        results[name] = {
            'pre_adversarial': {
                'accuracy': pre_acc,
                'predictions': pre_pred,
                'report': classification_report(y_test_pre, pre_pred)
            },
            'post_adversarial': {
                'accuracy': post_acc,
                'predictions': post_pred,
                'report': classification_report(y_test_post, post_pred)
            }
        }
        
        print(f"\n=== {name} Results ===")
        print(f"Pre-adversarial accuracy: {pre_acc:.4f}")
        print(f"Post-adversarial accuracy: {post_acc:.4f}")
        print(f"Accuracy drop: {pre_acc - post_acc:.4f}")
        
        print("\nPre-adversarial Classification Report:")
        print(results[name]['pre_adversarial']['report'])
        
        print("\nPost-adversarial Classification Report:")
        print(results[name]['post_adversarial']['report'])
        
        # # Plot confusion matrices
        # plt.figure(figsize=(15, 5))
        
        # plt.subplot(1, 2, 1)
        # sns.heatmap(confusion_matrix(y_test_pre, pre_pred), 
        #            annot=True, fmt='d', cmap='Blues')
        # plt.title(f'{name} Pre-Adversarial Confusion Matrix')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        
        # plt.subplot(1, 2, 2)
        # sns.heatmap(confusion_matrix(y_test_post, post_pred), 
        #            annot=True, fmt='d', cmap='Blues')
        # plt.title(f'{name} Post-Adversarial Confusion Matrix')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        
        # plt.tight_layout()
        # plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrices.png')
        # plt.close()
    
    return results, models

def evaluate_accuracy_drop(pre_adv_path, adv_folders, random_state=42):
    """
    Evaluate accuracy drop for multiple adversarial folders.
    """
    # Load pre-adversarial data
    X_pre = np.load(f'{pre_adv_path}/X_features.npy')
    y_pre = np.load(f'{pre_adv_path}/y_labels.npy')
    
    # Create train/test indices
    indices = np.arange(len(X_pre))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=random_state,
        stratify=y_pre
    )
    
    # Split pre-adversarial data
    X_train = X_pre[train_indices]
    y_train = y_pre[train_indices]
    X_test_pre = X_pre[test_indices]
    y_test_pre = y_pre[test_indices]
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state)
    }
    
    # Initialize results dictionary
    results = {model: [] for model in models.keys()}
    
    for folder in adv_folders:
        # Load post-adversarial data
        X_post = np.load(f'{folder}/X_features.npy')
        y_post = np.load(f'{folder}/y_labels.npy')
        
        # Use the same test indices for post-adversarial test samples
        X_test_post = X_post[test_indices]
        y_test_post = y_post[test_indices]
        
        for name, model in models.items():
            # Train model on pre-adversarial training data
            model.fit(X_train, y_train)
            
            # Evaluate on pre-adversarial and post-adversarial test sets
            pre_acc = accuracy_score(y_test_pre, model.predict(X_test_pre))
            post_acc = accuracy_score(y_test_post, model.predict(X_test_post))
            
            # Calculate accuracy drop
            accuracy_drop = pre_acc - post_acc
            results[name].append(accuracy_drop)
    
    return results

def plot_results(results, epsilons):
    """
    Plot accuracy drop for each model against epsilon values.
    """
    plt.figure(figsize=(10, 6))
    for model, drops in results.items():
        plt.plot(epsilons, drops, marker='o', label=model)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy Drop (%)')
    plt.title('Accuracy Drop vs. Epsilon for Models')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    # Paths to data
    pre_adv_path = 'embed_cifar10_test_images_by_class'
    adv_folders = [
        'embed_adversarial_cifar10_epsilon_0.001',
        'embed_adversarial_cifar10_epsilon_0.005',
        'embed_adversarial_cifar10_epsilon_0.01',
        'embed_adversarial_cifar10_epsilon_0.05',
        'embed_adversarial_cifar10_epsilon_0.1',
        'embed_adversarial_cifar10_epsilon_0.5'
    ]
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    
    # Evaluate accuracy drop
    results = evaluate_accuracy_drop(pre_adv_path, adv_folders)
    
    # Plot results
    plot_results(results, epsilons)

