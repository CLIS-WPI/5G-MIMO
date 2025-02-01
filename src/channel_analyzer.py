# channel_analyzer.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from config import CONFIG, CHANNEL_CONFIG
import seaborn as sns
from tqdm import tqdm

# Create results directory
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def analyze_all_datasets():
    """Analyze train, validation, and test datasets"""
    datasets = {
        "train": CONFIG["output_files"]["train"],
        "validation": CONFIG["output_files"]["validation"],
        "test": CONFIG["output_files"]["test"]
    }
    
    all_stats = {}  # Store statistics for comparative analysis
    
    for split_name, filepath in tqdm(datasets.items(), desc="Analyzing datasets"):
        print(f"\nAnalyzing {split_name} dataset...")
        
        # Create split-specific output filenames
        output_plot = os.path.join(RESULTS_DIR, f'channel_analysis_{split_name}.png')
        output_stats = os.path.join(RESULTS_DIR, f'channel_stats_{split_name}.txt')
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)
        os.makedirs(os.path.dirname(output_stats), exist_ok=True)
        
        # Analyze dataset and save results
        with open(output_stats, 'w') as stats_file:
            original_stdout = sys.stdout
            sys.stdout = stats_file
            
            try:
                stats = analyze_channel_dataset(filepath, output_plot)
                all_stats[split_name] = stats
            except Exception as e:
                print(f"Error analyzing {split_name} dataset: {str(e)}")
            finally:
                sys.stdout = original_stdout
        
        print(f"Analysis for {split_name} dataset complete!")
        print(f"- Plot saved to: {output_plot}")
        print(f"- Statistics saved to: {output_stats}")
    
    # Generate comparative analysis
    create_comparative_analysis(all_stats)
    create_comparative_plots(all_stats)

def analyze_channel_dataset(filename, plot_filename):
    """Analyze channel dataset and return statistics"""
    stats = {}
    
    with h5py.File(filename, 'r') as f:
        batch_keys = list(f.keys())
        print("\nDataset Structure:")
        sample_batch = f[batch_keys[0]]
        print_structure(sample_batch)
        
        # Initialize data storage
        data = {
            'channel_magnitude': [],
            'channel_phase': [],
            'snr': [],
            'sinr': [],
            'capacity': [],
            'condition_numbers': []
        }
        
        # Add shape validation
        expected_shapes = get_expected_shapes(sample_batch)
        
        # Collect data from all batches
        for batch_key in tqdm(batch_keys, desc="Processing batches"):
            batch = f[batch_key]
            process_batch(batch, data, expected_shapes)
        
        # Concatenate all data
        stats = concatenate_data(data)
        
        # Create visualization plots
        create_analysis_plots(stats, plot_filename)
        
        # Print statistics
        print_channel_statistics(stats)
        
        return stats

def process_batch(batch, data, expected_shapes):
    """Process a single batch of data"""
    if isinstance(batch['channel'], h5py.Dataset):
        channel_data = batch['channel'][:]
        data['channel_magnitude'].append(np.abs(channel_data))
        data['channel_phase'].append(np.angle(channel_data))
    
    if 'snr' in batch:
        data['snr'].append(batch['snr'][:])
    
    if 'metrics' in batch:
        process_metrics(batch['metrics'], data, expected_shapes)
    
    if 'characteristics' in batch:
        process_characteristics(batch['characteristics'], data)

def process_metrics(metrics, data, expected_shapes):
    """Process metrics data"""
    for metric_name in ['capacity', 'sinr']:
        if metric_name in metrics:
            metric_data = metrics[metric_name][:]
            if metric_data.shape == expected_shapes[metric_name]:
                data[metric_name].append(metric_data)
            else:
                print(f"Warning: Skipping {metric_name} data with unexpected shape {metric_data.shape}")

def process_characteristics(chars, data):
    """Process characteristics data"""
    if 'condition_number' in chars:
        data['condition_numbers'].append(chars['condition_number'][:])

def concatenate_data(data):
    """Concatenate all collected data"""
    stats = {}
    for key, values in data.items():
        if values:
            try:
                stats[key] = np.concatenate(values, axis=0)
            except Exception as e:
                print(f"Warning: Could not concatenate {key}: {str(e)}")
                stats[key] = None
    return stats

def create_analysis_plots(stats, plot_filename):
    """Create comprehensive analysis plots"""
    plt.figure(figsize=(20, 15))
    
    plots = {
        'channel_magnitude': ('Channel Magnitude Distribution', 'Magnitude', 'Density'),
        'channel_phase': ('Channel Phase Distribution', 'Phase (radians)', 'Density'),
        'snr': ('SNR Distribution', 'SNR (dB)', 'Count'),
        'capacity': ('Channel Capacity Distribution', 'Capacity (bits/s/Hz)', 'Count'),
        'condition_numbers': ('Condition Number Distribution', 'Condition Number', 'Count')
    }
    
    for i, (key, (title, xlabel, ylabel)) in enumerate(plots.items(), 1):
        plt.subplot(3, 2, i)
        if key in stats and stats[key] is not None:
            plt.hist(stats[key].flatten(), bins=50, density=True)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
    
    # Channel Response Heatmap
    if 'channel_magnitude' in stats and stats['channel_magnitude'] is not None:
        plt.subplot(3, 2, 6)
        sample_response = stats['channel_magnitude'][0, 0, 0]
        plt.imshow(sample_response, aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title('Channel Response')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('OFDM Symbol')
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparative_plots(all_stats):
    """Create comparative plots between splits"""
    metrics = ['channel_magnitude', 'snr', 'capacity']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for split_name, stats in all_stats.items():
            if stats[metric] is not None:
                sns.kdeplot(data=stats[metric].flatten(), label=split_name)
        plt.title(f'{metric.replace("_", " ").title()} Distribution Comparison')
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'comparative_{metric}.png'))
        plt.close()

def print_structure(group, indent=0):
    """Print HDF5 group structure"""
    for key in group.keys():
        item = group[key]
        indent_str = "  " * indent
        if isinstance(item, h5py.Dataset):
            print(f"{indent_str}{key} (Dataset): shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{indent_str}{key} (Group):")
            print_structure(item, indent + 1)

def get_expected_shapes(sample_batch):
    """Get expected shapes from sample batch"""
    return {
        'capacity': sample_batch['metrics/capacity'].shape,
        'sinr': sample_batch['metrics/sinr'].shape
    }

def print_channel_statistics(stats):
    """Print comprehensive channel statistics"""
    print("\nChannel Statistics Summary:")
    print("=" * 50)
    
    metrics = [
        ('Channel Magnitude', 'channel_magnitude'),
        ('Channel Phase', 'channel_phase'),
        ('SNR', 'snr'),
        ('Capacity', 'capacity'),
        ('Condition Numbers', 'condition_numbers')
    ]
    
    for title, key in metrics:
        if key in stats and stats[key] is not None:
            data = stats[key]
            print(f"\n{title}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Range: [{np.min(data):.4f}, {np.max(data):.4f}]")
            if key == 'condition_numbers':
                print(f"  Geometric Mean: {np.exp(np.mean(np.log(data))):.4f}")

def create_comparative_analysis(all_stats):
    """Create comparative analysis between splits"""
    output_file = os.path.join(RESULTS_DIR, 'comparative_analysis.txt')
    metrics = ['channel_magnitude', 'snr', 'capacity']
    
    with open(output_file, 'w') as f:
        f.write("Comparative Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics comparison
        for metric in metrics:
            f.write(f"\n{metric.upper()} Statistics:\n")
            f.write("-" * 30 + "\n")
            for split_name, stats in all_stats.items():
                if stats[metric] is not None:
                    data = stats[metric]
                    f.write(f"{split_name}:\n")
                    f.write(f"  Mean: {np.mean(data):.4f}\n")
                    f.write(f"  Std:  {np.std(data):.4f}\n")
                    f.write(f"  Range: [{np.min(data):.4f}, {np.max(data):.4f}]\n")
            
        # Cross-split correlations using sampling
        f.write("\nCross-Split Correlations (using sampled data):\n")
        f.write("-" * 30 + "\n")
        splits = list(all_stats.keys())
        
        # Find minimum size across all splits and metrics
        min_size = float('inf')
        for split in splits:
            for metric in metrics:
                if all_stats[split][metric] is not None:
                    min_size = min(min_size, all_stats[split][metric].size)
        
        sample_size = min(min_size, 1000000)  # Use at most 1M samples
        
        for i, split1 in enumerate(splits):
            for split2 in splits[i+1:]:
                for metric in metrics:
                    if (all_stats[split1][metric] is not None and 
                        all_stats[split2][metric] is not None):
                        # Sample data from both splits
                        data1 = all_stats[split1][metric].flatten()
                        data2 = all_stats[split2][metric].flatten()
                        
                        # Random sampling
                        np.random.seed(42)  # For reproducibility
                        idx1 = np.random.choice(data1.size, sample_size, replace=False)
                        idx2 = np.random.choice(data2.size, sample_size, replace=False)
                        
                        sampled1 = data1[idx1]
                        sampled2 = data2[idx2]
                        
                        corr = np.corrcoef(sampled1, sampled2)[0,1]
                        f.write(f"{split1}-{split2} {metric}: {corr:.4f}\n")

if __name__ == "__main__":
    print("Starting channel analysis...")
    analyze_all_datasets()
    print("\nAnalysis complete! Check 'results' directory for all analyses.")