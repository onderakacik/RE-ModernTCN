# A script to visualize the ERF. ADAPTED FROM:
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# --------------------------------------------------------'
import argparse
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

#   Set figure parameters
large = 24; med = 24; small = 24
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 4),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
try:
    plt.style.use('seaborn-whitegrid')  # For older versions
except:
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions
sns.set_style("white")
plt.rc('font', **{'family': 'Times New Roman'})
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser('Script for analyzing the ERF', add_help=False)
parser.add_argument('--source', default='temp.npy', type=str, help='path to the contribution score matrix (.npy file)')
parser.add_argument('--heatmap_save', default='heatmap.png', type=str, help='where to save the heatmap')
args = parser.parse_args()

def heatmap(data, camp='RdYlGn', figsize=(32, 2), ax=None, save_path=None):
    # Debug prints
    print(f"Data shape before processing: {data.shape}")
    print(f"Data range before processing: [{np.min(data)}, {np.max(data)}]")
    print(f"Any NaN values: {np.any(np.isnan(data))}")
    
    fig = plt.figure(figsize=figsize, dpi=40)
    
    # Handle NaN values
    if np.any(np.isnan(data)):
        print("Warning: NaN values found in data. Replacing with zeros.")
        data = np.nan_to_num(data, nan=0.0)
    
    # For 1D sequence data, reshape to 2D for visualization
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    
    # Create custom colormap: black (0) -> red -> white (1)
    colors = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]  # black, red, white
    custom_cmap = LinearSegmentedColormap.from_list('custom_red', colors)
    
    ax = plt.gca() if ax is None else ax
    ax = sns.heatmap(data,
                     xticklabels=40,  # Show ticks every 40 positions
                     yticklabels=False,
                     cmap=custom_cmap,
                     center=None,
                     vmin=0,
                     vmax=1,
                     annot=False,
                     cbar=True,
                     cbar_kws={"orientation": "vertical"})
    
    # Set x-axis ticks every 40 positions
    seq_length = data.shape[1]
    xticks = np.arange(0, seq_length, 40)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_title('')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Heatmap saved at {save_path}')
    plt.close()

def get_rectangle(data, thresh):
    """Calculate the rectangular area that contains thresh% of the total contribution for 1D data."""
    total_sum = np.sum(data)
    if total_sum == 0:
        return None
    
    # For 1D data, we expand from center point until we reach the threshold
    center = len(data) // 2
    for i in range(1, len(data) // 2):
        window = data[center - i:center + i + 1]
        if np.sum(window) / total_sum > thresh:
            return 2 * i + 1, (2 * i + 1) / len(data)
    return None

def analyze_erf(args):
    # Load data
    data = np.load(args.source)
    print("Original data range:")
    print(f"Max value: {np.max(data)}")
    print(f"Min value: {np.min(data)}")
    
    # Handle 1D data
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    
    # Following equation (2) from the paper, adapted for 1D:
    data = np.log10(data + 1)
    # Rescale to [0,1] for comparability
    data = data / np.max(data)
    
    print('\n======================= ERF Analysis =====================')
    # Analyze different energy thresholds
    thresholds = [0.2, 0.3, 0.5, 0.99]
    for thresh in thresholds:
        result = get_rectangle(data.flatten(), thresh)  # Flatten for 1D analysis
        if result:
            size, ratio = result
            print(f'Threshold {thresh*100:.0f}%: size={size}, area ratio={ratio:.4f}')
    
    # Generate heatmap
    heatmap(data, save_path=args.heatmap_save)

if __name__ == '__main__':
    analyze_erf(args)
