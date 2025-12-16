import torch
import pandas as pd
import numpy as np
from pathlib import Path
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

def load_features_averaged(feature_dir, sample_id):
    """
    Load features.pt for a single sample and average over dimension 0
    Args:
        feature_dir: Feature directory path
        sample_id: Sample ID
    Returns:
        averaged_feature: 1024-dimensional feature vector
    """
    sample_path = Path(feature_dir) / sample_id / "features.pt"
    
    if not sample_path.exists():
        print(f"Warning: {sample_path} does not exist")
        return None
    
    try:
        # Load features
        features = torch.load(sample_path, map_location='cpu')
        # Average over dimension 0 to get 1024-dimensional feature
        averaged_feature = features.mean(dim=0)
        return averaged_feature.numpy()
    except Exception as e:
        print(f"Error loading {sample_path}: {e}")
        return None

def load_labels_from_kfold(kfold_dir):
    """
    Load labels for all samples from kfold directory
    Args:
        kfold_dir: kfold directory path
    Returns:
        labels_dict: {sample_id: label_name}
    """
    kfold_path = Path(kfold_dir)
    labels_dict = {}
    
    # Read all train and val files to get complete label information
    for csv_file in kfold_path.glob("*.csv"):
        if csv_file.name == "label.csv":
            continue
        
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            slide_id = row['slide_id']
            if slide_id not in labels_dict:
                # Find the column (label) with value 1
                label_cols = [col for col in df.columns if col != 'slide_id']
                for col in label_cols:
                    if row[col] == 1:
                        labels_dict[slide_id] = col
                        break
    
    return labels_dict

def load_dataset(feature_dir, kfold_dir, dataset_name):
    """
    Load all sample features and labels for a dataset
    Args:
        feature_dir: Feature directory
        kfold_dir: kfold label directory
        dataset_name: Dataset name
    Returns:
        features_list: Feature list
        labels_list: Label list
        sample_ids: Sample ID list
    """
    print(f"\nLoading {dataset_name} dataset...")
    
    # Load labels
    labels_dict = load_labels_from_kfold(kfold_dir)
    print(f"Found {len(labels_dict)} samples with labels")
    
    features_list = []
    labels_list = []
    sample_ids = []
    
    # Iterate through all samples
    feature_path = Path(feature_dir)
    sample_dirs = [d for d in feature_path.iterdir() if d.is_dir()]
    
    for sample_dir in tqdm(sample_dirs, desc=f"Loading {dataset_name} features"):
        sample_id = sample_dir.name
        
        # Only process samples with labels
        if sample_id in labels_dict:
            feature = load_features_averaged(feature_dir, sample_id)
            
            if feature is not None:
                features_list.append(feature)
                labels_list.append(labels_dict[sample_id])
                sample_ids.append(f"{dataset_name}_{sample_id}")
    
    print(f"{dataset_name} dataset loaded: {len(features_list)} samples")
    
    return features_list, labels_list, sample_ids

def plot_3d_umap(embedding, labels, sample_ids, dataset_sources):

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    color_map = {'DHMC': 'red', 'CLWD': 'blue'}
    marker_map = {'DHMC': 'o', 'CLWD': 'D'}
    for dataset in ['DHMC', 'CLWD']:
        mask = np.array(dataset_sources) == dataset
        if np.sum(mask) > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=color_map[dataset],
                label=f"{dataset} (n={np.sum(mask)})",
                alpha=0.95,
                s=60,
                edgecolors='k',
                linewidths=0.8,
                marker=marker_map[dataset]
            )
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_zlabel('UMAP 3', fontsize=12)
    ax.set_title('3D UMAP Visualization of DHMC (red) and CLWD (blue)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = "umap_3d_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n3D UMAP visualization saved to: {output_path}")
    plt.close(fig)

def plot_2d_umap(embedding, labels, sample_ids, dataset_sources):

    fig, ax = plt.subplots(figsize=(10, 8))
    color_map = {'DHMC': 'red', 'CLWD': 'blue'}
    marker_map = {'DHMC': 'o', 'CLWD': 'D'}
    for dataset in ['DHMC', 'CLWD']:
        mask = np.array(dataset_sources) == dataset
        if np.sum(mask) > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color_map[dataset],
                label=f"{dataset} (n={np.sum(mask)})",
                alpha=0.95,
                s=60,
                edgecolors='k',
                linewidths=0.8,
                marker=marker_map[dataset]
            )
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('2D UMAP Visualization of DHMC (red) and CLWD (blue)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = "umap_2d_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D UMAP visualization saved to: {output_path}")
    plt.close(fig)

def main():
    # Dataset paths
    dhmc_feature_dir = r"data\DHMC\feature-256-50"
    dhmc_kfold_dir = r"labels\DHMC\kfold"
    clwd_feature_dir = r"data\CLWD\feature-256-50"
    clwd_kfold_dir = r"labels\CLWD\kfold"

    # Load DHMC dataset
    dhmc_features, dhmc_labels, dhmc_ids = load_dataset(
        dhmc_feature_dir, dhmc_kfold_dir, "DHMC"
    )
    # Load CLWD dataset
    clwd_features, clwd_labels, clwd_ids = load_dataset(
        clwd_feature_dir, clwd_kfold_dir, "CLWD"
    )

    # Merge both datasets
    all_features = np.vstack(dhmc_features + clwd_features)
    all_labels = dhmc_labels + clwd_labels
    all_sample_ids = dhmc_ids + clwd_ids
    all_datasets = ['DHMC'] * len(dhmc_features) + ['CLWD'] * len(clwd_features)

    print(f"\nMerged dataset:")
    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimensions: {all_features.shape}")
    print(f"Label distribution:")
    for label in sorted(set(all_labels)):
        count = all_labels.count(label)
        print(f"  {label}: {count}")

    # UMAP dimension reduction to 3D
    print("\nStarting UMAP 3D reduction...")
    reducer_3d = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
    )
    embedding_3d = reducer_3d.fit_transform(all_features)
    print(f"UMAP 3D reduction completed! Reduced dimensions: {embedding_3d.shape}")
    plot_3d_umap(embedding_3d, all_labels, all_sample_ids, all_datasets)
  

    # UMAP dimension reduction to 2D
    print("\nStarting UMAP 2D reduction...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
    )
    embedding_2d = reducer_2d.fit_transform(all_features)

    print(f"UMAP 2D reduction completed! Reduced dimensions: {embedding_2d.shape}")
    plot_2d_umap(embedding_2d, all_labels, all_sample_ids, all_datasets)



if __name__ == "__main__":
    main()