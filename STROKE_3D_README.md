# 3D Stroke Clustering Visualization
## Pattern Recognition - Stock Market Stroke Analysis

Complete toolkit for extracting stroke-like patterns from stock data and visualizing clusters in 3D space.

---

## 🎯 What This Does

Extracts "strokes" (price movement patterns) from stock data and clusters them using K-Means, then creates beautiful 3D visualizations showing:
- Each stroke as a point in 3D PCA space
- Clusters colored by group
- Cluster centroids as large red X markers
- Perfect cubic aspect ratio (no distortion)

---

## 📦 Two Versions Available

### Version 1: Standalone Visualizer (Quick)
**Use when:** You already have `X_scaled`, `labels`, and `kmeans` from clustering

**File:** `visualize_3d_clusters.py`

**Usage:**
```python
# After running kmeans_clustering_analysis.py, you have:
# - X_scaled
# - labels  
# - kmeans

# Simply run:
python visualize_3d_clusters.py
```

### Version 2: Complete Pipeline (Full Power)
**Use when:** You want the full workflow from raw CSV to 3D visualization

**File:** `stroke_clustering_pipeline.py`

**Usage:**
```python
# Automatically finds stock CSVs, extracts strokes, clusters, and visualizes
python stroke_clustering_pipeline.py
```

---

## 🚀 Quick Start

### Option A: Full Pipeline (Recommended)

```bash
# 1. Generate stock data (if you haven't)
python stock_collector_free.py

# 2. Run complete pipeline
python stroke_clustering_pipeline.py
```

**That's it!** You'll get:
- Stroke extraction from stock data
- Automatic optimal K detection
- 3D cluster visualization
- All results saved to CSV

### Option B: Just Visualization

```bash
# 1. Run clustering first
python kmeans_clustering_analysis.py

# 2. Then visualize in 3D
python visualize_3d_clusters.py
```

---

## 📊 What Are "Strokes"?

Strokes are price movement patterns between turning points (peaks and troughs).

**Example:**
```
Stock Price:  ↗ ↗ ↗ (peak) ↘ ↘ ↘ (trough) ↗ ↗
Strokes:      |-- Stroke 1 --|  |-- Stroke 2 --|
```

**Each stroke is characterized by:**
- `duration` - How many time periods
- `amplitude` - Price change (end - start)
- `direction` - Upward (+1) or downward (-1)
- `volatility` - Standard deviation of prices
- `price_range` - Max - min price
- `curvature` - How curved vs straight
- `volume_change` - Change in trading volume
- `avg_volume` - Average volume during stroke
- `volume_volatility` - Volume variation
- `stroke_length` - Euclidean distance in price-time space

---

## 🎨 3D Visualization Features

### What You'll See

**4 Key Elements:**
1. **Colored Points** - Each stroke colored by cluster
2. **Red X Markers** - Cluster centroids (centers)
3. **Cubic Space** - Equal scaling on all axes (no distortion)
4. **PCA Labels** - Shows % variance explained by each axis

### Reading the Plot

**Clusters Close Together** = Similar stroke patterns
**Clusters Far Apart** = Distinct stroke behaviors
**Centroid Position** = "Average" stroke in that cluster

### Example Interpretation

```
Cluster 0 (Blue): High volatility, large amplitude strokes
  → Rapid price movements, high trading activity

Cluster 1 (Green): Moderate volatility, medium duration
  → Steady trending periods

Cluster 2 (Yellow): Low volatility, short duration
  → Consolidation, sideways movement
```

---

## 📁 Output Files

### From Full Pipeline

```
extracted_strokes.csv              # All extracted strokes with features
strokes_with_clusters.csv          # Strokes + cluster assignments
stock_strokes_3d_K[N].png         # 3D visualization
```

### CSV File Structure

**extracted_strokes.csv:**
```csv
symbol,duration,amplitude,abs_amplitude,direction,volatility,price_range,curvature,...
AAPL,12,3.45,3.45,1,0.87,4.23,0.56,...
AAPL,8,-2.31,2.31,-1,0.62,2.89,0.34,...
GOOGL,15,8.92,8.92,1,1.23,9.45,0.78,...
```

**strokes_with_clusters.csv:**
```csv
symbol,duration,amplitude,...,cluster
AAPL,12,3.45,...,0
AAPL,8,-2.31,...,1
GOOGL,15,8.92,...,0
```

---

## 🔧 Configuration Options

### Standalone Visualizer

Edit these variables in `visualize_3d_clusters.py`:

```python
# Visual settings
FIGURE_SIZE = (14, 10)          # Plot size
POINT_SIZE = 50                 # Stroke point size
POINT_ALPHA = 0.6               # Point transparency
CENTROID_SIZE = 300             # Centroid marker size
CENTROID_MARKER = 'X'           # Centroid shape
CENTROID_COLOR = 'red'          # Centroid color

# Viewing angle
ELEVATION = 25                  # Vertical angle
AZIMUTH = 35                    # Horizontal rotation
```

### Full Pipeline

Edit in `stroke_clustering_pipeline.py`:

```python
# Stroke extraction
min_stroke_length = 5           # Minimum stroke duration

# Clustering
n_clusters = None               # Auto-detect optimal K
# n_clusters = 3                # Or set manually

# Number of stocks to process
for stock_file in stock_files[:3]:  # Change 3 to process more
```

---

## 🎓 Understanding PCA in 3D

### What is PCA?

**PCA (Principal Component Analysis)** reduces 11 stroke features down to 3 dimensions for visualization.

**The 3 axes (PC1, PC2, PC3) are:**
- **PC1** - Direction of maximum variance (most important)
- **PC2** - Second most important direction
- **PC3** - Third most important direction

### Explained Variance

Example output:
```
PC1 explains 42.3% of variance
PC2 explains 28.7% of variance  
PC3 explains 15.2% of variance
Total: 86.2%
```

**Interpretation:** 86.2% of stroke variation is captured in the 3D plot.

Higher total = Better representation of original data

---

## 🔍 Stroke Extraction Algorithm

### Step-by-Step Process

**1. Detect Turning Points**
```python
# Find local maxima (peaks)
# Find local minima (troughs)
# Create segments between consecutive turning points
```

**2. Calculate Features for Each Segment**
```python
duration = end_idx - start_idx
amplitude = price[end] - price[start]
volatility = std(prices)
curvature = deviation_from_straight_line
# ... and 7 more features
```

**3. Filter Short Strokes**
```python
# Remove strokes shorter than min_stroke_length
# Keeps only meaningful patterns
```

### Customizing Detection

**More sensitive (more strokes):**
```python
min_stroke_length = 3           # Shorter minimum
distance = 2                    # Closer turning points
```

**Less sensitive (fewer, longer strokes):**
```python
min_stroke_length = 10          # Longer minimum
distance = 5                    # More spaced turning points
```

---

## 📊 Cluster Interpretation Guide

### Typical Cluster Patterns

**Cluster 0: "Breakouts"**
- High amplitude
- High volatility
- Medium-long duration
- High volume change
- **Interpretation:** Strong trending moves

**Cluster 1: "Consolidations"**
- Low amplitude
- Low volatility
- Short duration
- Low volume
- **Interpretation:** Sideways movement, indecision

**Cluster 2: "Corrections"**
- Medium amplitude
- Medium volatility
- Short-medium duration
- Negative direction bias
- **Interpretation:** Pullbacks, profit-taking

**Cluster 3: "Trends"**
- Medium-high amplitude
- Low curvature (straight)
- Long duration
- Steady volume
- **Interpretation:** Sustained directional moves

---

## 🎯 Use Cases

### Trading Strategy Development
```python
# Identify which strokes precede big moves
pre_breakout_strokes = strokes_df[strokes_df['cluster'] == 0]

# Analyze characteristics
avg_duration = pre_breakout_strokes['duration'].mean()
avg_volume = pre_breakout_strokes['avg_volume'].mean()

# Build entry signals
```

### Market Regime Classification
```python
# Group by time period
strokes_df['date'] = ...
regime_by_period = strokes_df.groupby('date')['cluster'].mode()

# Identify when market was in each regime
```

### Stock Comparison
```python
# Compare stroke patterns across stocks
for symbol in symbols:
    symbol_strokes = strokes_df[strokes_df['symbol'] == symbol]
    cluster_distribution = symbol_strokes['cluster'].value_counts()
    print(f"{symbol}: {cluster_distribution}")
```

### Pattern Recognition
```python
# Find similar historical patterns
current_stroke = extract_current_stroke()
current_features = calculate_features(current_stroke)
predicted_cluster = kmeans.predict([current_features])

# Look at historical strokes in same cluster
similar_strokes = strokes_df[strokes_df['cluster'] == predicted_cluster]
```

---

## 🔧 Troubleshooting

### "No stock CSV files found"
**Solution:**
```bash
python stock_collector_free.py
```

### "Not enough strokes extracted"
**Cause:** Stock data too short or min_stroke_length too high

**Solution:**
```python
# In stroke_clustering_pipeline.py, change:
extractor = StrokeExtractor(min_stroke_length=3)  # Was 5
```

### "Clusters overlap too much"
**Cause:** Features may not be discriminative

**Solution:**
- Try different K values
- Add more stocks for more data
- Check if features need scaling adjustments

### "PCA explains low variance (<60%)"
**Meaning:** 3D plot doesn't capture full complexity

**Not necessarily bad!** Just means original 11D space is complex.

**Solution:** Still useful for visualization, but don't over-interpret small separations.

---

## 📚 Technical Details

### Equal Aspect Ratio (Cubic Plot)

**Why it matters:** Without equal scaling, clusters can appear distorted.

**How it works:**
```python
# Calculate single range for all axes
max_range = (X_pca.max(axis=0) - X_pca.min(axis=0)).max() / 2.0

# Center on data midpoint
mid_x = (X_pca[:, 0].max() + X_pca[:, 0].min()) * 0.5
mid_y = (X_pca[:, 1].max() + X_pca[:, 1].min()) * 0.5
mid_z = (X_pca[:, 2].max() + X_pca[:, 2].min()) * 0.5

# Apply same range to all axes
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Set 1:1:1 box aspect
ax.set_box_aspect([1, 1, 1])
```

**Result:** True spatial relationships preserved!

---

## 🎨 Customization Ideas

### Different Color Schemes

```python
# Replace in code:
colors = plt.cm.viridis(...)      # Default

# Try:
colors = plt.cm.plasma(...)       # Purple-orange
colors = plt.cm.coolwarm(...)     # Blue-red
colors = plt.cm.tab10(...)        # Distinct colors
colors = plt.cm.Set3(...)         # Pastel
```

### Interactive 3D (Rotatable)

```python
# Add at end of script:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Instead of plt.show(), use:
plt.ion()  # Interactive mode
plt.show()

# Now you can rotate the plot with mouse!
```

### Multiple Views

```python
# Create 4 subplots with different angles
fig = plt.figure(figsize=(16, 16))

angles = [(25, 35), (25, 125), (25, 215), (25, 305)]

for i, (elev, azim) in enumerate(angles, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    # ... plotting code ...
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'View {i}: elev={elev}°, azim={azim}°')
```

---

## 🎓 Learning Objectives

By using this toolkit, you'll learn:

✅ How to extract patterns from time series data  
✅ Feature engineering for financial data  
✅ Dimensionality reduction with PCA  
✅ 3D visualization techniques  
✅ Cluster interpretation in domain context  
✅ End-to-end ML pipeline construction  

---

## 📞 Quick Reference

### Run Full Pipeline
```bash
python stroke_clustering_pipeline.py
```

### Run Just Visualization
```bash
python visualize_3d_clusters.py
```

### Change Number of Clusters
```python
# In stroke_clustering_pipeline.py:
n_clusters = 4  # Change from None or 3
```

### Process More Stocks
```python
# In stroke_clustering_pipeline.py:
for stock_file in stock_files[:5]:  # Change 3 to 5
```

---

**Ready to visualize your stock strokes in 3D? 🚀📊📈**

Run the pipeline and explore your clustering results!
