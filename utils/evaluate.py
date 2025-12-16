from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os


def save_predictions(score, label, save_dir='./'):
    """
    Save prediction scores and true labels to the specified directory
    
    Args:
        score: Prediction probability scores (numpy array)
        label: True labels (numpy array)
        save_dir: Directory to save files
        kfold: Current fold number
        filename_prefix: Filename prefix
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save files
    score_path = os.path.join(save_dir, "scores.npy")
    label_path = os.path.join(save_dir, "labels.npy")

    np.save(score_path, score)
    np.save(label_path, label)


def evaluation(score, label):
    y_true = np.argmax(label, axis=1)
    y_pred = np.argmax(score, axis=1)
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(label, score, multi_class='ovr', average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return {
        'ACC': acc * 100,
        'AUC': auc * 100,
        'Recall': recall * 100,
        'Precision': precision * 100,
        'F1': f1 * 100
    }


def plot_evaluation_curves(y_true, y_score, class_names, save_dir='./', kfold=0):
    """
    Plot confusion matrix, ROC curves, and PR curves
    
    Args:
        y_true: True labels (one-hot encoded or label indices)
        y_score: Prediction probabilities (probability for each class)
        class_names: List of class names
        save_dir: Directory to save plots
    """
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # If y_true is one-hot encoded, convert to label indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        y_true_binary = y_true
    else:
        y_true_labels = y_true
        # Convert to one-hot encoding for multi-class ROC and PR curves
        y_true_binary = label_binarize(y_true, classes=range(len(class_names)))
        if y_true_binary.shape[1] == 1:  # Binary classification case
            y_true_binary = np.hstack([1-y_true_binary, y_true_binary])
    
    y_pred_labels = np.argmax(y_score, axis=1)
    n_classes = len(class_names)
    
    # Set figure style
    plt.style.use('default')
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown'])
    
    # 1. Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = metrics.confusion_matrix(y_true_labels, y_pred_labels)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation text
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_percent[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{c}\n({p:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (kfold={kfold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        if n_classes == 2 and i == 0:  # Skip first class for binary classification
            continue
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_binary[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curve for each class
    for i, color in zip(range(n_classes), colors):
        if n_classes == 2 and i == 0:  # Skip first class for binary classification
            continue
        plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Multi-class Classification (kfold={kfold})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot PR curves (Precision-Recall curves)
    plt.figure(figsize=(12, 8))
    
    # Calculate PR curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        if n_classes == 2 and i == 0:  # Skip first class for binary classification
            continue
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true_binary[:, i], y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_true_binary[:, i], y_score[:, i])
    
    # Plot PR curve for each class
    for i, color in zip(range(n_classes), colors):
        if n_classes == 2 and i == 0:  # Skip first class for binary classification
            continue
        plt.plot(recall[i], precision[i], color=color, linewidth=2,
                label=f'{class_names[i]} (AP = {average_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for Multi-class Classification (kfold={kfold})')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return calculated metrics
    results = {
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'confusion_matrix': cm
    }
    
    return results
