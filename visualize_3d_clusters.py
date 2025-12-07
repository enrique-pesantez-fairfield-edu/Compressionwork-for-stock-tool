"""
3D K-Means Cluster Visualization
Standalone version - assumes you have X_scaled, labels, and kmeans already loaded
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# ============================================================================
# CONFIGURATION
# ============================================================================

# Assumes these variables exist in your workspace:
# - X_scaled: standardized feature matrix (n_samples, n_features)
# - labels: cluster assignments array (n_samples,)
# - kmeans: trained K-Means model with .cluster_centers_ attribute

# Number of clusters (automatically detected from labels)
n_clusters = len(np.unique(labels))

# Visualization settings
FIGURE_SIZE = (14, 10)
POINT_SIZE = 50
POINT_ALPHA = 0.6
CENTROID_SIZE = 300
CENTROID_MARKER = 'X'
CENTROID_COLOR = 'red'
CENTROID_EDGECOLOR = 'black'
CENTROID_LINEWIDTH = 2

ELEVATION = 25
AZIMUTH = 35

# Color palette for clusters
COLORS = plt.cm.viridis(np.linspace(0, 1, n_clusters))

# ============================================================================
# STEP 1: PCA TRANSFORMATION
# ============================================================================

print("="*60)
print("3D CLUSTER VISUALIZATION")
print("="*60)

print(f"\nPerforming PCA...")
print(f"  Input shape: {X_scaled.shape}")
print(f"  Number of clusters: {n_clusters}")

# Perform PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Transform centroids to PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Print explained variance
explained_var = pca.explained_variance_ratio_
print(f"\n✓ PCA Complete")
print(f"  PC1 explains {explained_var[0]*100:.2f}% of variance")
print(f"  PC2 explains {explained_var[1]*100:.2f}% of variance")
print(f"  PC3 explains {explained_var[2]*100:.2f}% of variance")
print(f"  Total: {sum(explained_var)*100:.2f}%")

# ============================================================================
# STEP 2: CREATE 3D PLOT
# ============================================================================

print(f"\nCreating 3D visualization...")

fig = plt.figure(figsize=FIGURE_SIZE)
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster
for cluster_id in range(n_clusters):
    # Get points in this cluster
    cluster_mask = labels == cluster_id
    cluster_points = X_pca[cluster_mask]
    
    # Plot cluster points
    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        cluster_points[:, 2],
        c=[COLORS[cluster_id]],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label=f'Cluster {cluster_id}',
        edgecolors='black',
        linewidths=0.5
    )

# Plot centroids
ax.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    centroids_pca[:, 2],
    c=CENTROID_COLOR,
    s=CENTROID_SIZE,
    marker=CENTROID_MARKER,
    edgecolors=CENTROID_EDGECOLOR,
    linewidths=CENTROID_LINEWIDTH,
    label='Centroids',
    zorder=10
)

# ============================================================================
# STEP 3: ENFORCE EQUAL SCALING (CUBIC PLOT)
# ============================================================================

print(f"  Setting equal aspect ratio...")

# Calculate the range for equal scaling
max_range = (X_pca.max(axis=0) - X_pca.min(axis=0)).max() / 2.0

# Calculate midpoints
mid_x = (X_pca[:, 0].max() + X_pca[:, 0].min()) * 0.5
mid_y = (X_pca[:, 1].max() + X_pca[:, 1].min()) * 0.5
mid_z = (X_pca[:, 2].max() + X_pca[:, 2].min()) * 0.5

# Set equal limits
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Set box aspect to 1:1:1
ax.set_box_aspect([1, 1, 1])

# ============================================================================
# STEP 4: LABELS AND FORMATTING
# ============================================================================

print(f"  Adding labels and legend...")

# Axis labels
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)', 
              fontsize=12, labelpad=10)
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)', 
              fontsize=12, labelpad=10)
ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}% var)', 
              fontsize=12, labelpad=10)

# Title
ax.set_title(f'3D K-Means Clustering Visualization (K={n_clusters})\n'
             f'PCA Projection (Total Variance Explained: {sum(explained_var)*100:.1f}%)',
             fontsize=14, fontweight='bold', pad=20)

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

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Set viewing angle
ax.view_init(elev=ELEVATION, azim=AZIMUTH)

# Adjust tick label padding
ax.tick_params(axis='x', pad=5)
ax.tick_params(axis='y', pad=5)
ax.tick_params(axis='z', pad=5)

# ============================================================================
# STEP 5: DISPLAY AND SAVE
# ============================================================================

plt.tight_layout()

# Save figure
output_filename = f'cluster_3d_visualization_K{n_clusters}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_filename}")

# Display
plt.show()

print(f"\n{'='*60}")
print("VISUALIZATION COMPLETE")
print(f"{'='*60}")
print(f"\nCluster Statistics:")
for cluster_id in range(n_clusters):
    count = np.sum(labels == cluster_id)
    percentage = (count / len(labels)) * 100
    print(f"  Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

print(f"\nCentroid locations in PCA space:")
for cluster_id in range(n_clusters):
    centroid = centroids_pca[cluster_id]
    print(f"  Cluster {cluster_id}: "
          f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
