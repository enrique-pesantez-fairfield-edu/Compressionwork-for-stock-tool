"""
Complete Stock Stroke Analysis Pipeline
Full end-to-end: CSV reading → Feature extraction → K-Means clustering → 3D visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class StrokeExtractor:
    """Extract stroke-like patterns from stock price data"""
    
    def __init__(self, min_stroke_length=5):
        self.min_stroke_length = min_stroke_length
        
    def detect_turning_points(self, prices):
        """Detect local maxima and minima"""
        # Find peaks and troughs
        peaks, _ = signal.find_peaks(prices, distance=3)
        troughs, _ = signal.find_peaks(-prices, distance=3)
        
        # Combine and sort
        turning_points = np.sort(np.concatenate([peaks, troughs, [0, len(prices)-1]]))
        
        return turning_points
    
    def extract_strokes(self, df, symbol):
        """
        Extract stroke features from stock data
        
        Parameters:
        - df: DataFrame with stock data (must have 'close' and 'volume' columns)
        - symbol: Stock symbol for labeling
        
        Returns:
        - strokes_df: DataFrame with stroke features
        """
        prices = df['close'].values
        volumes = df['volume'].values
        
        # Detect turning points
        turning_points = self.detect_turning_points(prices)
        
        strokes = []
        
        # Extract strokes between consecutive turning points
        for i in range(len(turning_points) - 1):
            start_idx = turning_points[i]
            end_idx = turning_points[i + 1]
            
            if end_idx - start_idx < self.min_stroke_length:
                continue
            
            # Extract stroke segment
            stroke_prices = prices[start_idx:end_idx+1]
            stroke_volumes = volumes[start_idx:end_idx+1]
            
            # Calculate features
            duration = end_idx - start_idx
            amplitude = stroke_prices[-1] - stroke_prices[0]
            abs_amplitude = abs(amplitude)
            direction = 1 if amplitude > 0 else -1
            
            # Volatility (std of prices)
            volatility = np.std(stroke_prices)
            
            # Price range
            price_range = np.max(stroke_prices) - np.min(stroke_prices)
            
            # Curvature (deviation from straight line)
            linear_interpolation = np.linspace(stroke_prices[0], stroke_prices[-1], len(stroke_prices))
            curvature = np.mean(np.abs(stroke_prices - linear_interpolation))
            
            # Volume features
            volume_change = stroke_volumes[-1] - stroke_volumes[0]
            avg_volume = np.mean(stroke_volumes)
            volume_volatility = np.std(stroke_volumes)
            
            # Stroke length (Euclidean distance in price-time space)
            # Normalize time to price scale
            time_normalized = np.arange(len(stroke_prices)) * (price_range / duration) if duration > 0 else np.arange(len(stroke_prices))
            stroke_length = np.sum(np.sqrt(np.diff(time_normalized)**2 + np.diff(stroke_prices)**2))
            
            strokes.append({
                'symbol': symbol,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': duration,
                'amplitude': amplitude,
                'abs_amplitude': abs_amplitude,
                'direction': direction,
                'volatility': volatility,
                'price_range': price_range,
                'curvature': curvature,
                'volume_change': volume_change,
                'avg_volume': avg_volume,
                'volume_volatility': volume_volatility,
                'stroke_length': stroke_length
            })
        
        return pd.DataFrame(strokes)


class StrokeClusterer:
    """Cluster strokes using K-Means and visualize in 3D"""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        
    def prepare_features(self, strokes_df):
        """
        Prepare feature matrix from strokes
        
        Parameters:
        - strokes_df: DataFrame with stroke features
        
        Returns:
        - X: Feature matrix
        - feature_names: List of feature names
        """
        feature_cols = [
            'duration', 'amplitude', 'abs_amplitude', 'direction',
            'volatility', 'price_range', 'curvature', 'volume_change',
            'avg_volume', 'volume_volatility', 'stroke_length'
        ]
        
        X = strokes_df[feature_cols].values
        
        return X, feature_cols
    
    def find_optimal_k(self, X_scaled, k_range=range(2, 11)):
        """Find optimal number of clusters"""
        print(f"\n{'='*60}")
        print("Finding Optimal K")
        print(f"{'='*60}")
        
        results = []
        
        for k in k_range:
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
        
        results_df = pd.DataFrame(results)
        
        # Recommend based on silhouette score
        best_k = results_df.loc[results_df['Silhouette'].idxmax(), 'K']
        print(f"\n✓ Recommended K: {int(best_k)} (highest Silhouette score)")
        
        return results_df, int(best_k)
    
    def cluster_strokes(self, strokes_df, n_clusters=None):
        """
        Cluster strokes using K-Means
        
        Parameters:
        - strokes_df: DataFrame with stroke features
        - n_clusters: Number of clusters (if None, will find optimal)
        
        Returns:
        - X_scaled: Scaled feature matrix
        - labels: Cluster assignments
        - strokes_with_clusters: DataFrame with cluster labels
        """
        print(f"\n{'='*60}")
        print("STROKE CLUSTERING ANALYSIS")
        print(f"{'='*60}")
        
        # Prepare features
        X, feature_names = self.prepare_features(strokes_df)
        print(f"\n✓ Extracted {X.shape[1]} features from {X.shape[0]} strokes")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Features normalized")
        
        # Find optimal K if not specified
        if n_clusters is None:
            optimization_results, optimal_k = self.find_optimal_k(X_scaled)
            n_clusters = optimal_k
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
        Create 3D visualization of clusters
        
        Parameters:
        - X_scaled: Scaled feature matrix
        - labels: Cluster assignments
        - title_suffix: Additional text for title
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
        
        # Enforce equal scaling (cubic plot)
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
    """Complete pipeline: Load data → Extract strokes → Cluster → Visualize"""
    
    print("="*60)
    print("STOCK STROKE CLUSTERING PIPELINE")
    print("="*60)
    
    # ========================================================================
    # STEP 1: Load Stock Data
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("STEP 1: Loading Stock Data")
    print(f"{'='*60}")
    
    # Look for stock CSV files
    import glob
    stock_files = glob.glob("*_with_indicators.csv") + glob.glob("*_stock_data.csv")
    
    if not stock_files:
        print("\n✗ No stock CSV files found!")
        print("  Looking for: *_with_indicators.csv or *_stock_data.csv")
        print("\n  Run stock_collector_free.py first to generate data.")
        return
    
    print(f"\n✓ Found {len(stock_files)} stock file(s)")
    
    # Use first file or combine multiple
    all_strokes = []
    
    for stock_file in stock_files[:3]:  # Process up to 3 stocks
        print(f"\nProcessing: {stock_file}")
        
        df = pd.read_csv(stock_file)
        symbol = stock_file.split('_')[0]
        
        print(f"  Loaded {len(df)} rows")
        
        # Extract strokes
        extractor = StrokeExtractor(min_stroke_length=5)
        strokes_df = extractor.extract_strokes(df, symbol)
        
        print(f"  ✓ Extracted {len(strokes_df)} strokes")
        
        all_strokes.append(strokes_df)
    
    # Combine all strokes
    combined_strokes = pd.concat(all_strokes, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Total strokes extracted: {len(combined_strokes)}")
    print(f"{'='*60}")
    
    # Save extracted strokes
    combined_strokes.to_csv('extracted_strokes.csv', index=False)
    print(f"\n✓ Saved: extracted_strokes.csv")
    
    # ========================================================================
    # STEP 2: Cluster Strokes
    # ========================================================================
    
    clusterer = StrokeClusterer()
    
    # Let it find optimal K or specify manually
    X_scaled, labels, strokes_with_clusters, feature_names = clusterer.cluster_strokes(
        combined_strokes,
        n_clusters=None  # Auto-detect optimal K
        # n_clusters=3   # Or specify manually
    )
    
    # Save clustered strokes
    strokes_with_clusters.to_csv('strokes_with_clusters.csv', index=False)
    print(f"\n✓ Saved: strokes_with_clusters.csv")
    
    # ========================================================================
    # STEP 3: 3D Visualization
    # ========================================================================
    
    symbols_str = ", ".join(combined_strokes['symbol'].unique())
    X_pca, centroids_pca = clusterer.visualize_3d(
        X_scaled, 
        labels,
        title_suffix=f"Stocks: {symbols_str}"
    )
    
    # ========================================================================
    # STEP 4: Cluster Analysis
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("CLUSTER CHARACTERISTICS")
    print(f"{'='*60}")
    
    for cluster_id in range(clusterer.n_clusters):
        cluster_data = strokes_with_clusters[strokes_with_clusters['cluster'] == cluster_id]
        
        print(f"\nCluster {cluster_id} ({len(cluster_data)} strokes):")
        print(f"  Average duration: {cluster_data['duration'].mean():.1f} periods")
        print(f"  Average amplitude: {cluster_data['amplitude'].mean():.2f}")
        print(f"  Average volatility: {cluster_data['volatility'].mean():.2f}")
        print(f"  Direction: {cluster_data['direction'].mean():.2f} "
              f"({'upward' if cluster_data['direction'].mean() > 0 else 'downward'} trend)")
        
        # Most common symbol in this cluster
        top_symbol = cluster_data['symbol'].value_counts().iloc[0]
        top_symbol_name = cluster_data['symbol'].value_counts().index[0]
        print(f"  Most common stock: {top_symbol_name} ({top_symbol} strokes)")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    print(f"\n📁 Files Created:")
    print(f"  • extracted_strokes.csv")
    print(f"  • strokes_with_clusters.csv")
    print(f"  • stock_strokes_3d_K{clusterer.n_clusters}.png")
    
    return clusterer, strokes_with_clusters, X_scaled, labels


if __name__ == "__main__":
    # Install dependencies if needed
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
        clusterer, strokes_df, X_scaled, labels = main()
        print("\n🎉 Pipeline complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
