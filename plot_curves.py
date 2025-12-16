import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import os
import argparse

def plot_kfold_curves(save_dir, n_folds=5, class_names=None):
    """
    Plot average ROC curves, PR curves, and confusion matrices for K-fold cross-validation
    
    Parameters:
        save_dir: Directory to save results, each fold's results are saved in fold_{k} subdirectory
        n_folds: Number of folds, default is 5
        class_names: List of class names, default is None
    """
    
    # Store results for each fold
    all_scores = []
    all_labels = []
    
    # Read scores.npy and labels.npy for each fold
    print(f"Reading data from {save_dir}...")
    for k in range(n_folds):
        fold_dir = os.path.join(save_dir, f'fold_{k}')
        scores_path = os.path.join(fold_dir, 'scores.npy')
        labels_path = os.path.join(fold_dir, 'labels.npy')
        
        if os.path.exists(scores_path) and os.path.exists(labels_path):
            scores = np.load(scores_path)
            labels = np.load(labels_path)
            all_scores.append(scores)
            all_labels.append(labels)
            print(f"Fold {k}: scores shape {scores.shape}, labels shape {labels.shape}")
        else:
            print(f"Warning: Data files for Fold {k} do not exist")
    
    if len(all_scores) == 0:
        print("Error: No data files found")
        return
    
    # Determine number of classes
    n_classes = all_scores[0].shape[1] if len(all_scores[0].shape) > 1 else 2
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Create directory to save figures
    output_dir = os.path.join(save_dir, 'kfold_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Plot ROC curves ====================
    plot_roc_curves(all_scores, all_labels, n_classes, class_names, output_dir)
    
    # ==================== Plot PR curves ====================
    plot_pr_curves(all_scores, all_labels, n_classes, class_names, output_dir)
    
    # ==================== Plot average confusion matrix ====================
    plot_average_confusion_matrix(all_scores, all_labels, n_classes, class_names, output_dir)
    
    print(f"\nAll charts saved to: {output_dir}")


def plot_roc_curves(all_scores, all_labels, n_classes, class_names, output_dir):
    """Plot ROC curves for multi-class classification"""
    
    mean_fpr = np.linspace(0, 1, 100)
    
    # Create a single figure with average ROC curves for all classes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate average ROC curve for each class
    for class_idx in range(n_classes):
        tprs = []
        aucs = []
        
        # Calculate ROC curve for each fold
        for fold_idx, (scores, labels) in enumerate(zip(all_scores, all_labels)):
            # Get scores and labels for current class
            if len(scores.shape) > 1:
                y_score = scores[:, class_idx]
            else:
                y_score = scores
            
            # Convert labels to binary classification (current class vs others)
            if len(labels.shape) > 1:
                y_true = labels[:, class_idx]
            else:
                y_true = (labels == class_idx).astype(int)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            
            # Interpolate to fixed fpr points
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            
            # Calculate AUC
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        
        # Calculate average ROC curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        # Plot average ROC curve
        ax.plot(mean_fpr, mean_tpr, lw=2, alpha=0.8,
               label=f'{class_names[class_idx]} (AUC={mean_auc:.3f}±{std_auc:.3f})')
        
        # Calculate standard deviation and plot confidence interval
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.15)
    
    # Draw diagonal line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', 
           label='Chance', alpha=0.8)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Mean ROC Curves for All Classes', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_all_classes.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC curves saved")


def plot_pr_curves(all_scores, all_labels, n_classes, class_names, output_dir):
    """Plot PR curves (Precision-Recall) for multi-class classification"""
    
    # Create a single figure with average PR curves for all classes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Recall points for interpolation
    mean_recall = np.linspace(0, 1, 100)
    
    # Calculate average PR curve for each class
    for class_idx in range(n_classes):
        precisions_interp = []
        aps = []  # Average Precision scores
        
        # Calculate PR curve for each fold
        for fold_idx, (scores, labels) in enumerate(zip(all_scores, all_labels)):
            # Get scores and labels for current class
            if len(scores.shape) > 1:
                y_score = scores[:, class_idx]
            else:
                y_score = scores
            
            # Convert labels to binary classification
            if len(labels.shape) > 1:
                y_true = labels[:, class_idx]
            else:
                y_true = (labels == class_idx).astype(int)
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            
            # Calculate Average Precision
            ap = average_precision_score(y_true, y_score)
            aps.append(ap)
            
            # Reverse arrays for interpolation (recall from 0 to 1)
            precision = precision[::-1]
            recall = recall[::-1]
            
            # Interpolate to fixed recall points
            precision_interp = np.interp(mean_recall, recall, precision)
            precisions_interp.append(precision_interp)
        
        # Calculate average PR curve
        mean_precision = np.mean(precisions_interp, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)
        
        # Plot average PR curve
        ax.plot(mean_recall, mean_precision, lw=2, alpha=0.8,
               label=f'{class_names[class_idx]} (AP={mean_ap:.3f}±{std_ap:.3f})')
        
        # Calculate standard deviation and plot confidence interval
        std_precision = np.std(precisions_interp, axis=0)
        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall, precision_lower, precision_upper, alpha=0.15)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Mean Precision-Recall Curves for All Classes', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves_all_classes.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("PR curves saved")


def plot_average_confusion_matrix(all_scores, all_labels, n_classes, class_names, output_dir):
    """Plot average confusion matrix"""
    
    confusion_matrices = []
    
    # Calculate confusion matrix for each fold
    for fold_idx, (scores, labels) in enumerate(zip(all_scores, all_labels)):
        # Get predicted classes
        if len(scores.shape) > 1:
            y_pred = np.argmax(scores, axis=1)
        else:
            y_pred = (scores > 0.5).astype(int)
        
        # Get true labels
        if len(labels.shape) > 1:
            y_true = np.argmax(labels, axis=1)
        else:
            y_true = labels.astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        confusion_matrices.append(cm)
    
    # Calculate average confusion matrix
    mean_cm = np.mean(confusion_matrices, axis=0)
    std_cm = np.std(confusion_matrices, axis=0)
    
    # Calculate normalized average confusion matrix (row-normalized)
    mean_cm_normalized = mean_cm / mean_cm.sum(axis=1, keepdims=True)
    
    # Create a figure to display both raw values and normalized values
    fig, ax = plt.subplots(figsize=(max(10, n_classes*1.5), max(8, n_classes*1.2)))
    
    # Plot heatmap
    sns.heatmap(mean_cm, annot=False, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'}, 
                linewidths=0.5, linecolor='gray')
    
    # Manually add annotations: raw value on top, normalized percentage below
    for i in range(n_classes):
        for j in range(n_classes):
            # Raw value (integer or 1 decimal place)
            count_text = f'{mean_cm[i, j]:.1f}'
            # Normalized percentage
            percent_text = f'({mean_cm_normalized[i, j]:.1%})'
            
            # Combined text: value on top, percentage below
            combined_text = f'{count_text}\n{percent_text}'
            
            # Choose text color based on background color (white text for dark background)
            text_color = 'white' if mean_cm[i, j] > mean_cm.max() / 2 else 'black'
            
            ax.text(j + 0.5, i + 0.5, combined_text,
                   ha='center', va='center', 
                   color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Average Confusion Matrix (Count and Percentage)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_average.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save standard deviation confusion matrix
    fig, ax = plt.subplots(figsize=(max(10, n_classes*1.5), max(8, n_classes*1.2)))
    sns.heatmap(std_cm, annot=False, fmt='', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Std Dev'},
                linewidths=0.5, linecolor='gray')
    
    # Manually add annotations: display standard deviation values
    for i in range(n_classes):
        for j in range(n_classes):
            # Standard deviation value
            std_text = f'{std_cm[i, j]:.2f}'
            
            # Choose text color based on background color
            text_color = 'white' if std_cm[i, j] > std_cm.max() / 2 else 'black'
            
            ax.text(j + 0.5, i + 0.5, std_text,
                   ha='center', va='center', 
                   color=text_color, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix Standard Deviation', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_std.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Confusion matrices saved")
    
    # Print average confusion matrix statistics
    print("\n=== Average Confusion Matrix ===")
    print(mean_cm)
    print("\n=== Normalized Average Confusion Matrix ===")
    print(mean_cm_normalized)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot average curves for K-fold cross-validation')
    parser.add_argument('--save_dir', type=str, default=r"G:\LUAD-cohort3\results\luad\transmil",
                       help='Directory path to save results')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds, default is 5')
    parser.add_argument('--class_names', type=str, nargs='+', 
                       default=["papillary","lepidic","in situ","solid","micropapillary","cribriform","acinar"],
                       help='List of class names, e.g.: --class_names Class0 Class1')
    args = parser.parse_args()
    
    plot_kfold_curves(args.save_dir, args.n_folds, args.class_names)
