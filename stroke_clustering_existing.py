"""
3D Stroke Clustering Visualization
Modified to work with existing stroke CSV files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')
import glob
import os


class StrokeClusterer:
    """Cluster existing strokes and visualize in 3D"""
    
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        
    def load_stroke_files(self):
        """
        Load stroke CSV files from current directory
        Looks for: *_strokes.csv (NOT marked_data files)
        """
        print(f"\n{'='*60}")
        print("Loading Stroke Files")
        print(f"{'='*60}")
        
        # ONLY look for stroke files, NOT marked_data
        stroke_files = glob.glob("*_strokes.csv")
        
        # Explicitly exclude marked_data files
        stroke_files = [f for f in stroke_files if 'marked_data' not in f.lower()]
        
        if not stroke_files:
            print("\n✗ No stroke files found!")
            print("  Looking for: *_strokes.csv")
            print(f"\n  Current directory: {os.getcwd()}")
            print(f"  Files in directory:")
            for f in glob.glob("*.csv")[:10]:
                print(f"    - {f}")
            return None
        
        print(f"\n✓ Found {len(stroke_files)} stroke file(s):")
        for f in stroke_files:
            print(f"  - {f}")
        
        # Load and combine all stroke files
        all_strokes = []
        
        for file_path in stroke_files:
            try:
                print(f"\nLoading: {file_path}")
                df = pd.read_csv(file_path)
                
                # Extract symbol from filename
                symbol = file_path.split('_')[0].upper()
                
                # Add symbol column if not present
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                
                # CRITICAL FIX: Convert direction column from text to numeric
                if 'direction' in df.columns:
                    print(f"  Converting 'direction' column from text to numeric...")
                    df['direction'] = df['direction'].map({'up': 1, 'down': -1, 1: 1, -1: -1})
                
                print(f"  ✓ Loaded {len(df)} rows")
                print(f"  Columns ({len(df.columns)} total):")
                
                # Show first 15 columns with their types
                for i, col in enumerate(df.columns[:15], 1):
                    dtype = df[col].dtype
                    sample = df[col].iloc[0] if len(df) > 0 else "N/A"
                    print(f"    {i:2d}. {col:25s} (type: {str(dtype):10s}, sample: {str(sample)[:20]})")
                
                if len(df.columns) > 15:
                    print(f"    ... and {len(df.columns) - 15} more columns")
                
                all_strokes.append(df)
                
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")
                continue
        
        if not all_strokes:
            print("\n✗ Could not load any stroke files")
            return None
        
        # Combine all dataframes
        combined_strokes = pd.concat(all_strokes, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"Total Data Loaded")
        print(f"{'='*60}")
        print(f"Total rows: {len(combined_strokes)}")
        print(f"Total columns: {len(combined_strokes.columns)}")
        print(f"\nColumn list:")
        for i, col in enumerate(combined_strokes.columns, 1):
            print(f"  {i:2d}. {col}")
        
        return combined_strokes
    
    def prepare_features(self, strokes_df):
        """
        Prepare feature matrix from strokes DataFrame
        Automatically detects available numerical features
        """
        print(f"\n{'='*60}")
        print("Preparing Features")
        print(f"{'='*60}")
        
        # Target feature columns (from original spec)
        target_features = [
            'duration', 'amplitude', 'abs_amplitude', 'direction',
            'volatility', 'price_range', 'curvature', 'volume_change',
            'avg_volume', 'volume_volatility', 'stroke_length'
        ]
        
        # Find which target features exist in the data
        available_features = [col for col in target_features if col in strokes_df.columns]
        
        if not available_features:
            # If none of the target features exist, use all numerical columns
            print("  Target features not found, detecting numerical columns...")
            
            # Get all column names
            all_cols = strokes_df.columns.tolist()
            
            # Exclude obvious non-numeric patterns AND index columns
            exclude_patterns = ['index', 'idx', 'Unnamed', 'symbol', 'name', 'type', 
                              'date', 'time', 'id', 'datetime']
            
            # Test each column to see if it's truly numeric
            available_features = []
            for col in all_cols:
                # Skip if matches exclude pattern
                if any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                    continue
                
                # Try to convert to numeric
                try:
                    test_values = pd.to_numeric(strokes_df[col], errors='coerce')
                    # Check if we have at least some valid numeric values
                    if test_values.notna().sum() > len(strokes_df) * 0.5:  # At least 50% valid
                        available_features.append(col)
                except:
                    continue
        
        if not available_features:
            print("\n✗ No numerical features found!")
            print("\nAvailable columns in data:")
            for col in strokes_df.columns:
                sample_vals = strokes_df[col].head(3).tolist()
                print(f"  - {col:25s} (type: {str(strokes_df[col].dtype):10s})")
                print(f"    Sample values: {sample_vals}")
            raise ValueError("No numerical features found in stroke data!")
        
        print(f"\n✓ Selected {len(available_features)} features:")
        for i, col in enumerate(available_features, 1):
            print(f"  {i:2d}. {col}")
        
        # Extract and convert to numeric, forcing errors to NaN
        feature_data = strokes_df[available_features].copy()
        
        # Convert all columns to numeric
        print(f"\nConverting features to numeric...")
        for col in available_features:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
        
        # Now convert to numpy array (safe because all numeric or NaN)
        X = feature_data.values.astype(float)
        
        # Check for NaN values
        nan_mask = np.isnan(X)
        if nan_mask.any():
            nan_count = nan_mask.any(axis=1).sum()
            print(f"\n⚠ Warning: Found {nan_count} rows with NaN values")
            
            # Show which columns have the most NaNs
            print(f"\n  NaN count by column:")
            for i, col in enumerate(available_features):
                nan_in_col = nan_mask[:, i].sum()
                if nan_in_col > 0:
                    print(f"    {col:20s}: {nan_in_col} NaN values ({nan_in_col/len(X)*100:.1f}%)")
            
            valid_mask = ~nan_mask.any(axis=1)
            X = X[valid_mask]
            strokes_df = strokes_df.iloc[valid_mask].reset_index(drop=True)
            print(f"\n  Rows after removing NaN: {len(X)}")
        
        # Check if we have any data left
        if len(X) == 0:
            print(f"\n✗ ERROR: No valid data rows remaining after NaN removal!")
            print(f"\n  This usually means:")
            print(f"    - All columns contain non-numeric data")
            print(f"    - OR columns are empty")
            print(f"    - OR wrong columns were selected")
            print(f"\n  Original data had {len(strokes_df)} rows")
            print(f"  Please check your CSV files!")
            raise ValueError("No valid numeric data found!")
        
        print(f"\n✓ Feature matrix shape: {X.shape}")
        
        if X.shape[0] > 0:
            print(f"  Feature ranges:")
            for i, col in enumerate(available_features):
                print(f"    {col:20s}: [{X[:, i].min():10.2f}, {X[:, i].max():10.2f}]")
        
        return X, available_features, strokes_df
    
    def find_optimal_k(self, X_scaled, k_range=range(2, 11)):
        """Find optimal number of clusters"""
        print(f"\n{'='*60}")
        print("Finding Optimal K")
        print(f"{'='*60}")
        
        results = []
        
        for k in k_range:
            if k > len(X_scaled):
                print(f"K={k}: Skipped (more clusters than samples)")
                continue
                
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            
            results.append({
                'K': k,
                'Inertia': inertia,
                'Silhouette': silhouette,
                'Davies-Bouldin': db_score
            })
            
            print(f"K={k}: Silhouette={silhouette:.4f}, Inertia={inertia:.2f}, DB={db_score:.4f}")
        
        if not results:
            print("\n✗ Could not test any K values")
            return None, 2
        
        results_df = pd.DataFrame(results)
        
        # Recommend based on silhouette score
        best_k = results_df.loc[results_df['Silhouette'].idxmax(), 'K']
        print(f"\n✓ Recommended K: {int(best_k)} (highest Silhouette score)")
        
        return results_df, int(best_k)
    
    def cluster_strokes(self, strokes_df, n_clusters=None):
        """
        Cluster strokes using K-Means
        """
        print(f"\n{'='*60}")
        print("STROKE CLUSTERING ANALYSIS")
        print(f"{'='*60}")
        
        # Prepare features
        X, feature_names, strokes_df = self.prepare_features(strokes_df)
        
        if X.shape[0] < 2:
            print("\n✗ Not enough data points for clustering")
            return None, None, None, None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        print(f"\n✓ Features normalized (StandardScaler)")
        
        # Find optimal K if not specified
        if n_clusters is None:
            max_k = min(10, len(X_scaled) - 1)
            optimization_results, optimal_k = self.find_optimal_k(X_scaled, k_range=range(2, max_k + 1))
            
            if optimization_results is not None:
                n_clusters = optimal_k
            else:
                n_clusters = min(3, len(X_scaled) - 1)
            
            self.n_clusters = n_clusters
        else:
            self.n_clusters = n_clusters
        
        # Perform clustering
        print(f"\n{'='*60}")
        print(f"Clustering with K={n_clusters}")
        print(f"{'='*60}")
        
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        
        print(f"\n✓ Clustering complete")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {db_score:.4f}")
        
        # Add cluster labels to dataframe
        strokes_with_clusters = strokes_df.copy()
        strokes_with_clusters['cluster'] = labels
        
        # Print cluster sizes
        print(f"\nCluster Sizes:")
        for cluster_id in range(n_clusters):
            count = np.sum(labels == cluster_id)
            percentage = (count / len(labels)) * 100
            print(f"  Cluster {cluster_id}: {count} strokes ({percentage:.1f}%)")
        
        return X_scaled, labels, strokes_with_clusters, feature_names
    
    def visualize_3d(self, X_scaled, labels, title_suffix=""):
        """
        Create 3D visualization of clusters with perfect cubic aspect ratio
        """
        print(f"\n{'='*60}")
        print("Creating 3D Visualization")
        print(f"{'='*60}")
        
        # Perform PCA
        self.pca = PCA(n_components=3)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Transform centroids to PCA space
        centroids_pca = self.pca.transform(self.kmeans.cluster_centers_)
        
        # Print explained variance
        explained_var = self.pca.explained_variance_ratio_
        print(f"\n✓ PCA Complete")
        print(f"  PC1: {explained_var[0]*100:.2f}% variance")
        print(f"  PC2: {explained_var[1]*100:.2f}% variance")
        print(f"  PC3: {explained_var[2]*100:.2f}% variance")
        print(f"  Total: {sum(explained_var)*100:.2f}%")
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color palette
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        
        # Plot each cluster
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = X_pca[cluster_mask]
            
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2],
                c=[colors[cluster_id]],
                s=50,
                alpha=0.6,
                label=f'Cluster {cluster_id}',
                edgecolors='black',
                linewidths=0.5
            )
        
        # Plot centroids
        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            centroids_pca[:, 2],
            c='red',
            s=300,
            marker='X',
            edgecolors='black',
            linewidths=2,
            label='Centroids',
            zorder=10
        )
        
        # ENFORCE EQUAL SCALING (CUBIC PLOT)
        max_range = (X_pca.max(axis=0) - X_pca.min(axis=0)).max() / 2.0
        mid_x = (X_pca[:, 0].max() + X_pca[:, 0].min()) * 0.5
        mid_y = (X_pca[:, 1].max() + X_pca[:, 1].min()) * 0.5
        mid_z = (X_pca[:, 2].max() + X_pca[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect([1, 1, 1])
        
        # Labels and formatting
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)', 
                     fontsize=12, labelpad=10)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)', 
                     fontsize=12, labelpad=10)
        ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}% var)', 
                     fontsize=12, labelpad=10)
        
        # Title
        title = f'3D K-Means Clustering of Stock Strokes (K={self.n_clusters})\n'
        title += f'PCA Projection (Total Variance: {sum(explained_var)*100:.1f}%)'
        if title_suffix:
            title += f'\n{title_suffix}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        legend = ax.legend(
            loc='upper left',
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            title='Legend',
            title_fontsize=11
        )
        legend.get_frame().set_alpha(0.9)
        
        # Grid and viewing angle
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.view_init(elev=25, azim=35)
        
        # Tick padding
        ax.tick_params(axis='x', pad=5)
        ax.tick_params(axis='y', pad=5)
        ax.tick_params(axis='z', pad=5)
        
        plt.tight_layout()
        
        # Save
        filename = f'stock_strokes_3d_K{self.n_clusters}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {filename}")
        
        plt.show()
        
        return X_pca, centroids_pca


def main():
    """Complete pipeline: Load existing strokes → Cluster → Visualize"""
    
    print("="*60)
    print("STOCK STROKE 3D CLUSTERING PIPELINE")
    print("="*60)
    
    # Initialize clusterer
    clusterer = StrokeClusterer()
    
    # Load stroke files
    strokes_df = clusterer.load_stroke_files()
    
    if strokes_df is None:
        print("\n✗ Could not load stroke data")
        return None
    
    # Cluster strokes (auto-detect optimal K)
    result = clusterer.cluster_strokes(strokes_df, n_clusters=None)
    
    if result[0] is None:
        print("\n✗ Clustering failed")
        return None
    
    X_scaled, labels, strokes_with_clusters, feature_names = result
    
    # Save clustered strokes
    output_file = 'strokes_with_clusters.csv'
    strokes_with_clusters.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # 3D Visualization
    symbols = strokes_with_clusters['symbol'].unique() if 'symbol' in strokes_with_clusters.columns else []
    symbols_str = ", ".join(symbols) if len(symbols) > 0 else "All Stocks"
    
    X_pca, centroids_pca = clusterer.visualize_3d(
        X_scaled, 
        labels,
        title_suffix=f"Stocks: {symbols_str}"
    )
    
    # Cluster Analysis
    print(f"\n{'='*60}")
    print("CLUSTER CHARACTERISTICS")
    print(f"{'='*60}")
    
    # Analyze clusters based on available features
    for cluster_id in range(clusterer.n_clusters):
        cluster_data = strokes_with_clusters[strokes_with_clusters['cluster'] == cluster_id]
        
        print(f"\nCluster {cluster_id} ({len(cluster_data)} strokes):")
        
        # Show statistics for key features if they exist
        feature_stats = {}
        for feature in feature_names[:5]:  # Show first 5 features
            if feature in cluster_data.columns:
                mean_val = cluster_data[feature].mean()
                feature_stats[feature] = mean_val
                print(f"  {feature:20s}: {mean_val:.2f}")
        
        # Most common symbol if available
        if 'symbol' in cluster_data.columns:
            top_symbol_count = cluster_data['symbol'].value_counts().iloc[0]
            top_symbol_name = cluster_data['symbol'].value_counts().index[0]
            print(f"  Most common stock: {top_symbol_name} ({top_symbol_count} strokes)")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    print(f"\n📁 Files Created:")
    print(f"  • strokes_with_clusters.csv")
    print(f"  • stock_strokes_3d_K{clusterer.n_clusters}.png")
    
    return clusterer, strokes_with_clusters, X_scaled, labels


if __name__ == "__main__":
    # Check for required packages
    try:
        import sklearn
        import scipy
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "scikit-learn", "scipy"])
        print("✓ Packages installed!\n")
    
    # Run pipeline
    try:
        result = main()
        if result is not None:
            clusterer, strokes_df, X_scaled, labels = result
            print("\n🎉 Pipeline complete!")
        else:
            print("\n⚠ Pipeline completed with warnings - check output above")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()