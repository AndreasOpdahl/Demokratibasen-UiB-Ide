#!/usr/bin/env python3
"""
Learning Curve Plotter for Fine-tuning Results

Reads trainer_state.json from a checkpoint folder and plots:
1. Training loss curve (actual values + smooth trend using Savitzky-Golay filter)
2. Evaluation rougeLsum scores (interpolated with cubic spline, actual points as dots)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def load_trainer_state(checkpoint_folder):
    """Load trainer_state.json from the specified checkpoint folder."""
    trainer_state_path = Path(checkpoint_folder) / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print(f"Error: {trainer_state_path} does not exist")
        sys.exit(1)
    
    with open(trainer_state_path, 'r') as f:
        return json.load(f)


def extract_metrics(log_history):
    """Extract loss and rougeLsum metrics from log_history."""
    loss_steps = []
    loss_values = []
    rouge_steps = []
    rouge_values = []
    
    for entry in log_history:
        step = entry.get('step')
        
        # Extract training loss
        if 'loss' in entry and step is not None:
            loss_steps.append(step)
            loss_values.append(entry['loss'])
        
        # Extract evaluation rougeLsum
        if 'eval_rougeLsum' in entry and step is not None:
            rouge_steps.append(step)
            rouge_values.append(entry['eval_rougeLsum'])
    
    return (np.array(loss_steps), np.array(loss_values),
            np.array(rouge_steps), np.array(rouge_values))


def interpolate_metrics(steps, values):
    """Create cubic spline interpolation for the given metric."""
    if len(steps) < 4:
        # Need at least 4 points for cubic spline, fall back to linear
        print(f"Warning: Only {len(steps)} points available, using linear interpolation")
        return None
    
    cs = CubicSpline(steps, values)
    
    # Create dense array for smooth plotting
    steps_dense = np.linspace(steps[0], steps[-1], 1000)
    values_interpolated = cs(steps_dense)
    
    return steps_dense, values_interpolated


def compute_smooth_trend(values, window_length=51, polyorder=3):
    """
    Compute a smooth trend curve using Savitzky-Golay filter.
    
    Args:
        values: Array of values to smooth
        window_length: The length of the filter window (must be odd and > polyorder)
        polyorder: The order of the polynomial used to fit the samples
    
    Returns:
        Smoothed values
    """
    if len(values) < window_length:
        # Adjust window length if we don't have enough data points
        window_length = len(values) if len(values) % 2 == 1 else len(values) - 1
        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
    
    try:
        smoothed = savgol_filter(values, window_length, polyorder)
        return smoothed
    except Exception as e:
        print(f"Warning: Could not compute smooth trend: {e}")
        return values


def plot_learning_curves(loss_steps, loss_values, rouge_steps, rouge_values, checkpoint_folder, output_path=None):
    """Create dual-axis plot with loss and rougeLsum curves."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot loss on primary y-axis (left)
    color_loss = 'tab:blue'
    color_loss_trend = 'darkblue'
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', color=color_loss, fontsize=12)
    
    # Plot actual loss values
    ax1.plot(loss_steps, loss_values, color=color_loss, 
            linewidth=1, alpha=0.4, label='Loss (actual)')
    
    # Compute and plot smooth trend
    loss_smooth = compute_smooth_trend(loss_values, window_length=51, polyorder=3)
    ax1.plot(loss_steps, loss_smooth, color=color_loss_trend, 
            linewidth=3, label='Loss (trend)', zorder=3)
    
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis (right) for rougeLsum
    ax2 = ax1.twinx()
    color_rouge = 'tab:orange'
    ax2.set_ylabel('ROUGE-Lsum Score', color=color_rouge, fontsize=12)
    
    # Interpolate rougeLsum
    rouge_interp = interpolate_metrics(rouge_steps, rouge_values)
    
    if rouge_interp is not None:
        # Plot interpolated curve
        ax2.plot(rouge_interp[0], rouge_interp[1], color=color_rouge, 
                linewidth=2, alpha=0.7, label='ROUGE-Lsum (interpolated)')
        # Plot actual measurements as dots
        ax2.scatter(rouge_steps, rouge_values, color=color_rouge, 
                   s=80, zorder=5, edgecolors='black', linewidths=1,
                   label='ROUGE-Lsum (actual)')
    else:
        ax2.plot(rouge_steps, rouge_values, color=color_rouge, 
                linewidth=2, marker='o', label='ROUGE-Lsum')
    
    ax2.tick_params(axis='y', labelcolor=color_rouge)
    
    # Add title and legends
    plt.title('Training Loss and ROUGE-Lsum Learning Curves', fontsize=14, fontweight='bold')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Tight layout for better spacing
    fig.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_path = Path(checkpoint_folder) / "learning_curve.png"
    else:
        output_path = Path(output_path)
    
    # Try to save plot, fallback to current directory if permission denied
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    except PermissionError:
        fallback_path = Path.cwd() / "learning_curve.png"
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(f"Permission denied for {output_path}")
        print(f"Plot saved to: {fallback_path}")
    
    # Display plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot learning curves from trainer_state.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_curve.py ../models/gemma_finetuned/checkpoint-6000
  python learning_curve.py ../models/gemma_finetuned/checkpoint-6000 -o my_plot.png
  python learning_curve.py path/to/checkpoint
        """
    )
    parser.add_argument(
        'checkpoint_folder',
        type=str,
        help='Path to checkpoint folder containing trainer_state.json'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: checkpoint_folder/learning_curve.png)'
    )
    
    args = parser.parse_args()
    
    # Load trainer state
    print(f"Loading trainer state from: {args.checkpoint_folder}")
    trainer_state = load_trainer_state(args.checkpoint_folder)
    
    # Extract metrics
    print("Extracting metrics from log history...")
    loss_steps, loss_values, rouge_steps, rouge_values = extract_metrics(
        trainer_state['log_history']
    )
    
    print(f"Found {len(loss_steps)} loss measurements")
    print(f"Found {len(rouge_steps)} ROUGE-Lsum measurements")
    
    if len(loss_steps) == 0 and len(rouge_steps) == 0:
        print("Error: No metrics found in trainer_state.json")
        sys.exit(1)
    
    # Plot learning curves
    print("Creating plot...")
    plot_learning_curves(loss_steps, loss_values, rouge_steps, rouge_values, 
                        args.checkpoint_folder, output_path=args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()

