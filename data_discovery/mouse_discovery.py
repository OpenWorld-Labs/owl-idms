import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import from owl_idms
sys.path.append(str(Path(__file__).parent.parent))
from owl_idms.data.cod_datasets import get_cod_paths, normalise_mouse


def analyze_mouse_distributions(root: str = "/home/shared/cod_data/", sample_size: int = 1000000):
    """Analyze mouse delta distributions to find better normalization strategies."""
    paths = get_cod_paths(root)
    print(f"Found {len(paths)} data files")
    
    all_mouse_deltas = []
    
    # Sample mouse delta data from multiple files
    sample_count = 0
    for i, (_, m_path, _) in enumerate(paths):
        if sample_count >= sample_size:
            break
            
        try:
            mouse_deltas = torch.load(m_path, map_location="cpu", mmap=True)
            
            # Take a random subset from this file
            if len(mouse_deltas) > 100:
                start_idx = np.random.randint(0, len(mouse_deltas) - 100)
                subset_size = min(100, sample_size - sample_count)
                mouse_subset = mouse_deltas[start_idx:start_idx + subset_size]
                all_mouse_deltas.append(mouse_subset)
                
                sample_count += len(mouse_subset)
                
        except Exception as e:
            print(f"Error loading {m_path}: {e}")
            continue
    
    if not all_mouse_deltas:
        print("No mouse data loaded!")
        return
    
    # Concatenate all delta data
    delta_data = torch.cat(all_mouse_deltas, dim=0)
    
    print(f"Loaded {len(delta_data)} mouse delta samples")
    print(f"Delta data shape: {delta_data.shape}")
    
    return analyze_and_plot(delta_data)


def analyze_and_plot(delta_data):
    """Analyze mouse delta data and compare normalization strategies."""
    
    # Convert to numpy for easier analysis
    delta_np = delta_data.float().numpy()
    
    # Assume delta data is [x, y] deltas
    if delta_np.shape[1] >= 2:
        delta_x, delta_y = delta_np[:, 0], delta_np[:, 1]
    else:
        print("Mouse delta data doesn't have expected shape")
        return
    
    # Calculate statistics
    print("\n=== MOUSE DELTA STATISTICS ===")
    print(f"Delta X range: [{delta_x.min():.2f}, {delta_x.max():.2f}]")
    print(f"Delta Y range: [{delta_y.min():.2f}, {delta_y.max():.2f}]")
    print(f"Delta X std: {delta_x.std():.2f}")
    print(f"Delta Y std: {delta_y.std():.2f}")
    print(f"Delta X mean: {delta_x.mean():.2f}")
    print(f"Delta Y mean: {delta_y.mean():.2f}")
    
    # Calculate percentiles for deltas
    delta_magnitudes = np.sqrt(delta_x**2 + delta_y**2)
    percentiles = [50, 90, 95, 99, 99.9]
    print(f"\nDelta magnitude percentiles:")
    for p in percentiles:
        print(f"  {p}%: {np.percentile(delta_magnitudes, p):.2f}")
    
    # Test different normalization strategies
    print("\n=== NORMALIZATION COMPARISON ===")
    
    # Standard normalization (z-score)
    delta_x_norm = (delta_x - delta_x.mean()) / delta_x.std()
    delta_y_norm = (delta_y - delta_y.mean()) / delta_y.std()
    
    # Min-max normalization
    delta_x_minmax = (delta_x - delta_x.min()) / (delta_x.max() - delta_x.min())
    delta_y_minmax = (delta_y - delta_y.min()) / (delta_y.max() - delta_y.min())
    
    # Log1p transformation (for handling large movements)
    delta_x_log1p = np.sign(delta_x) * np.log1p(np.abs(delta_x))
    delta_y_log1p = np.sign(delta_y) * np.log1p(np.abs(delta_y))
    
    # Robust scaling (using IQR)
    q75_x, q25_x = np.percentile(delta_x, [75, 25])
    q75_y, q25_y = np.percentile(delta_y, [75, 25])
    iqr_x, iqr_y = q75_x - q25_x, q75_y - q25_y
    delta_x_robust = (delta_x - np.median(delta_x)) / iqr_x if iqr_x > 0 else delta_x
    delta_y_robust = (delta_y - np.median(delta_y)) / iqr_y if iqr_y > 0 else delta_y
    
    # Clipped z-score (clip outliers before normalizing)
    delta_x_clipped = np.clip(delta_x, np.percentile(delta_x, 1), np.percentile(delta_x, 99))
    delta_y_clipped = np.clip(delta_y, np.percentile(delta_y, 1), np.percentile(delta_y, 99))
    delta_x_clipped_norm = (delta_x_clipped - delta_x_clipped.mean()) / delta_x_clipped.std()
    delta_y_clipped_norm = (delta_y_clipped - delta_y_clipped.mean()) / delta_y_clipped.std()
    
    # COD normalise_mouse function (from cod_datasets)
    delta_cod_normalized = []
    for dx, dy in zip(delta_x, delta_y):
        delta_tensor = torch.tensor([dx, dy], dtype=torch.float32)
        normalized_delta = normalise_mouse(delta_tensor).numpy()
        delta_cod_normalized.append(normalized_delta)
    delta_cod_normalized = np.array(delta_cod_normalized)
    delta_x_cod, delta_y_cod = delta_cod_normalized[:, 0], delta_cod_normalized[:, 1]
    
    methods = {
        'Original': (delta_x, delta_y),
        'Z-score': (delta_x_norm, delta_y_norm),
        'Min-Max': (delta_x_minmax, delta_y_minmax),
        'Log1p': (delta_x_log1p, delta_y_log1p),
        'Robust (IQR)': (delta_x_robust, delta_y_robust),
        'Clipped Z-score': (delta_x_clipped_norm, delta_y_clipped_norm),
        'COD normalise_mouse': (delta_x_cod, delta_y_cod)
    }
    
    for name, (dx, dy) in methods.items():
        print(f"\n{name}:")
        print(f"  X range: [{dx.min():.3f}, {dx.max():.3f}], std: {dx.std():.3f}")
        print(f"  Y range: [{dy.min():.3f}, {dy.max():.3f}], std: {dy.std():.3f}")
        magnitude = np.sqrt(dx**2 + dy**2)
        print(f"  Magnitude 99%: {np.percentile(magnitude, 99):.3f}")
    
    # Create visualizations
    create_plots(methods, delta_magnitudes)
    
    return methods


def create_plots(methods, original_magnitudes):
    """Create visualization plots for the different normalization methods."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Mouse Delta Normalization Analysis', fontsize=16)
    
    # Plot 1: Original delta distributions
    ax = axes[0, 0]
    dx_orig, dy_orig = methods['Original']
    ax.hist2d(dx_orig, dy_orig, bins=100, alpha=0.7)
    ax.set_title('Original Delta Distribution')
    ax.set_xlabel('Delta X')
    ax.set_ylabel('Delta Y')
    
    # Plot 2: Log1p transformed
    ax = axes[0, 1]
    dx_log, dy_log = methods['Log1p']
    ax.hist2d(dx_log, dy_log, bins=100, alpha=0.7)
    ax.set_title('Log1p Transformed Deltas')
    ax.set_xlabel('Log1p(Delta X)')
    ax.set_ylabel('Log1p(Delta Y)')
    
    # Plot 3: Magnitude distribution comparison
    ax = axes[1, 0]
    dx_orig, dy_orig = methods['Original']
    dx_log, dy_log = methods['Log1p']
    dx_robust, dy_robust = methods['Robust (IQR)']
    dx_cod, dy_cod = methods['COD normalise_mouse']
    
    mag_orig = np.sqrt(dx_orig**2 + dy_orig**2)
    mag_log = np.sqrt(dx_log**2 + dy_log**2)
    mag_robust = np.sqrt(dx_robust**2 + dy_robust**2)
    mag_cod = np.sqrt(dx_cod**2 + dy_cod**2)
    
    ax.hist(mag_orig, bins=100, alpha=0.5, label='Original', density=True)
    ax.hist(mag_log, bins=100, alpha=0.5, label='Log1p', density=True)
    ax.hist(mag_robust, bins=100, alpha=0.5, label='Robust', density=True)
    ax.hist(mag_cod, bins=100, alpha=0.5, label='COD normalise_mouse', density=True)
    ax.set_xlabel('Delta Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Delta Magnitude Distributions')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 4: Percentile comparison
    ax = axes[1, 1]
    percentiles = np.arange(50, 100, 1)
    methods_to_compare = ['Original', 'Log1p', 'Robust (IQR)', 'Clipped Z-score', 'COD normalise_mouse']
    
    for method_name in methods_to_compare:
        dx, dy = methods[method_name]
        magnitudes = np.sqrt(dx**2 + dy**2)
        percentile_values = [np.percentile(magnitudes, p) for p in percentiles]
        ax.plot(percentiles, percentile_values, label=method_name, marker='o', markersize=2)
    
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Delta Magnitude')
    ax.set_title('Delta Magnitude Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Box plots for different methods
    ax = axes[2, 0]
    magnitude_data = []
    labels = []
    for name, (dx, dy) in methods.items():
        if name != 'Original':  # Skip original to avoid scale issues
            magnitude_data.append(np.sqrt(dx**2 + dy**2))
            labels.append(name)
    
    ax.boxplot(magnitude_data, labels=labels)
    ax.set_ylabel('Delta Magnitude')
    ax.set_title('Delta Magnitude Distributions (Box Plot)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 6: Outlier analysis
    ax = axes[2, 1]
    dx_orig, dy_orig = methods['Original']
    outlier_threshold = np.percentile(original_magnitudes, 99)
    
    outlier_mask = original_magnitudes > outlier_threshold
    normal_mask = ~outlier_mask
    
    ax.scatter(dx_orig[normal_mask], dy_orig[normal_mask], alpha=0.1, s=1, label='Normal', c='blue')
    ax.scatter(dx_orig[outlier_mask], dy_orig[outlier_mask], alpha=0.7, s=10, label='Outliers (>99%)', c='red')
    ax.set_xlabel('Delta X')
    ax.set_ylabel('Delta Y')
    ax.set_title('Outlier Analysis (>99% percentile)')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent / 'mouse_normalization_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Starting mouse delta analysis...")
    methods = analyze_mouse_distributions(sample_size=5000)
    
    print("\n=== RECOMMENDATIONS ===")
    print("Based on the analysis:")
    print("1. Log1p transformation helps with handling large movements while preserving small ones")
    print("2. Robust scaling (IQR) is less sensitive to outliers than standard z-score")
    print("3. Clipped z-score can help if there are extreme outliers")
    print("4. Consider the distribution shape when choosing normalization strategy")