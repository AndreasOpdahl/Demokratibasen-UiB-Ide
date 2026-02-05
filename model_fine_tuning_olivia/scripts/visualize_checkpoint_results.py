"""
Visualize checkpoint evaluation results from all_eval_results/ directory.

This script loads all checkpoint evaluation JSON files and creates comprehensive
visualizations showing training progress across metrics.

Usage:
    # Basic visualization (generates HTML and PNG files)
    python visualize_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp \
        --output_dir visualizations/gemma-2-9b
    
    # With WandB logging (optional)
    python visualize_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp \
        --output_dir visualizations/gemma-2-9b \
        --wandb_project lm-evaluation \
        --wandb_run_name gemma-2-9b-visualization
    
    # Compare multiple models
    python visualize_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp models/gemma-7b-apptainer-fsdp \
        --output_dir visualizations/comparison \
        --compare_models
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available. Install with: pip install plotly")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available. Install with: pip install wandb")


def extract_checkpoint_step(filename: str) -> Optional[int]:
    """Extract checkpoint step number from filename."""
    match = re.search(r'checkpoint-(\d+)-eval-results\.json', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint_results(model_dir: str) -> Dict[int, Dict]:
    """Load all checkpoint evaluation results from all_eval_results/ directory.
    
    Returns:
        Dictionary mapping checkpoint step -> evaluation results
    """
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    if not os.path.exists(all_eval_results_dir):
        print(f"Warning: all_eval_results directory not found: {all_eval_results_dir}")
        return {}
    
    results = {}
    pattern = os.path.join(all_eval_results_dir, "checkpoint-*-eval-results.json")
    
    for filepath in glob.glob(pattern):
        try:
            step = extract_checkpoint_step(os.path.basename(filepath))
            if step is None:
                continue
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[step] = data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading {filepath}: {e}")
            continue
    
    return results


def extract_metrics(results: Dict[int, Dict]) -> Dict[str, List[Tuple[int, float]]]:
    """Extract metrics from results, organized by category.
    
    Returns:
        Dictionary mapping metric_name -> [(step, value), ...]
    """
    metrics = {}
    
    for step, data in sorted(results.items()):
        # ROUGE metrics
        for rouge_type in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            key = f'eval_{rouge_type}'
            if key in data:
                if rouge_type not in metrics:
                    metrics[rouge_type] = []
                metrics[rouge_type].append((step, data[key]))
        
        # Extended metrics - Reference-based (BERTScore)
        if 'eval_reference_bertscore_f1_mean' in data:
            if 'bertscore_f1' not in metrics:
                metrics['bertscore_f1'] = []
            metrics['bertscore_f1'].append((step, data['eval_reference_bertscore_f1_mean']))
        
        # Extended metrics - Hygiene
        if 'eval_hygiene_mean_compression_ratio' in data:
            if 'compression_ratio' not in metrics:
                metrics['compression_ratio'] = []
            metrics['compression_ratio'].append((step, data['eval_hygiene_mean_compression_ratio']))
        
        if 'eval_hygiene_mean_rep_3gram' in data:
            if 'repetition_3gram' not in metrics:
                metrics['repetition_3gram'] = []
            metrics['repetition_3gram'].append((step, data['eval_hygiene_mean_rep_3gram']))
        
        if 'eval_hygiene_ratio_ends_with_punct' in data:
            if 'ends_with_punct' not in metrics:
                metrics['ends_with_punct'] = []
            metrics['ends_with_punct'].append((step, data['eval_hygiene_ratio_ends_with_punct']))
        
        # Extended metrics - Faithfulness (nested dict)
        # Support both nested dict (new format) and flattened keys (backward compatibility)
        if 'eval_faithfulness' in data and isinstance(data['eval_faithfulness'], dict):
            faithfulness = data['eval_faithfulness']
            if 'mean_entailment_score' in faithfulness:
                if 'entailment_score' not in metrics:
                    metrics['entailment_score'] = []
                metrics['entailment_score'].append((step, faithfulness['mean_entailment_score']))
            
            if 'mean_ratio_outliers' in faithfulness:
                if 'outlier_rate' not in metrics:
                    metrics['outlier_rate'] = []
                metrics['outlier_rate'].append((step, faithfulness['mean_ratio_outliers']))
        # Backward compatibility: check for flattened keys
        elif 'eval_faithfulness_mean_entailment_score' in data:
            if 'entailment_score' not in metrics:
                metrics['entailment_score'] = []
            metrics['entailment_score'].append((step, data['eval_faithfulness_mean_entailment_score']))
        
        if 'eval_faithfulness_mean_ratio_outliers' in data:
            if 'outlier_rate' not in metrics:
                metrics['outlier_rate'] = []
            metrics['outlier_rate'].append((step, data['eval_faithfulness_mean_ratio_outliers']))
        elif 'eval_faithfulness_mean_outlier_rate' in data:  # Legacy key name
            if 'outlier_rate' not in metrics:
                metrics['outlier_rate'] = []
            metrics['outlier_rate'].append((step, data['eval_faithfulness_mean_outlier_rate']))
    
    return metrics


def create_matplotlib_plots(metrics: Dict[str, List[Tuple[int, float]]], 
                           output_dir: str, model_name: str):
    """Create matplotlib plots for all metrics."""
    if not HAS_MATPLOTLIB:
        print("Skipping matplotlib plots (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # ROUGE metrics
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    if any(m in metrics for m in rouge_metrics):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'ROUGE Metrics - {model_name}', fontsize=16, fontweight='bold')
        
        for idx, rouge_type in enumerate(rouge_metrics):
            ax = axes[idx // 2, idx % 2]
            if rouge_type in metrics:
                steps, values = zip(*metrics[rouge_type])
                ax.plot(steps, values, marker='o', linewidth=2, markersize=6)
                ax.set_title(f'ROUGE-{rouge_type.upper()}', fontweight='bold')
                ax.set_xlabel('Checkpoint Step')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'ROUGE-{rouge_type.upper()}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rouge_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/rouge_metrics.png")
    
    # Extended metrics - Reference-based
    if 'bertscore_f1' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps, values = zip(*metrics['bertscore_f1'])
        ax.plot(steps, values, marker='o', linewidth=2, markersize=6, color='green')
        ax.set_title(f'BERTScore F1 - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint Step')
        ax.set_ylabel('BERTScore F1')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bertscore.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/bertscore.png")
    
    # Extended metrics - Hygiene
    hygiene_metrics = ['compression_ratio', 'repetition_3gram', 'ends_with_punct']
    if any(m in metrics for m in hygiene_metrics):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Hygiene Metrics - {model_name}', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(hygiene_metrics):
            ax = axes[idx]
            if metric in metrics:
                steps, values = zip(*metrics[metric])
                ax.plot(steps, values, marker='o', linewidth=2, markersize=6)
                ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
                ax.set_xlabel('Checkpoint Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                if metric == 'ends_with_punct':
                    ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hygiene_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/hygiene_metrics.png")
    
    # Extended metrics - Faithfulness
    if 'entailment_score' in metrics or 'outlier_rate' in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Faithfulness Metrics - {model_name}', fontsize=16, fontweight='bold')
        
        if 'entailment_score' in metrics:
            steps, values = zip(*metrics['entailment_score'])
            axes[0].plot(steps, values, marker='o', linewidth=2, markersize=6, color='blue')
            axes[0].set_title('Mean Entailment Score', fontweight='bold')
            axes[0].set_xlabel('Checkpoint Step')
            axes[0].set_ylabel('Entailment Score')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1)
        else:
            axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Mean Entailment Score', fontweight='bold')
        
        if 'outlier_rate' in metrics:
            steps, values = zip(*metrics['outlier_rate'])
            axes[1].plot(steps, values, marker='o', linewidth=2, markersize=6, color='red')
            axes[1].set_title('Mean Outlier Rate', fontweight='bold')
            axes[1].set_xlabel('Checkpoint Step')
            axes[1].set_ylabel('Outlier Rate')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        else:
            axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Mean Outlier Rate', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'faithfulness_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/faithfulness_metrics.png")
    
    # Combined overview plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot ROUGE-Lsum (primary metric)
    if 'rougeLsum' in metrics:
        steps, values = zip(*metrics['rougeLsum'])
        ax.plot(steps, values, marker='o', linewidth=3, markersize=8, label='ROUGE-Lsum', color='blue')
    
    # Plot BERTScore if available
    if 'bertscore_f1' in metrics:
        steps, values = zip(*metrics['bertscore_f1'])
        # Normalize to same scale (multiply by 100 to match ROUGE percentage)
        values = [v * 100 for v in values]
        ax.plot(steps, values, marker='s', linewidth=2, markersize=6, label='BERTScore F1 (×100)', 
                color='green', linestyle='--')
    
    ax.set_title(f'Training Progress Overview - {model_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/overview.png")


def create_plotly_html(metrics: Dict[str, List[Tuple[int, float]]], 
                       output_dir: str, model_name: str):
    """Create interactive Plotly HTML visualization."""
    if not HAS_PLOTLY:
        print("Skipping Plotly HTML (plotly not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('ROUGE Metrics', 'BERTScore', 'Hygiene Metrics', 
                        'Faithfulness Metrics', 'Overview', ''),
        specs=[[{"colspan": 2}, None],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 2}, None]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # ROUGE metrics (row 1)
    rouge_colors = {'rouge1': 'blue', 'rouge2': 'green', 'rougeL': 'orange', 'rougeLsum': 'red'}
    for rouge_type in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        if rouge_type in metrics:
            steps, values = zip(*metrics[rouge_type])
            fig.add_trace(
                go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                          name=f'ROUGE-{rouge_type.upper()}', line=dict(color=rouge_colors[rouge_type], width=2)),
                row=1, col=1
            )
    
    # BERTScore (row 2, col 1)
    if 'bertscore_f1' in metrics:
        steps, values = zip(*metrics['bertscore_f1'])
        fig.add_trace(
            go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                      name='BERTScore F1', line=dict(color='purple', width=2)),
            row=2, col=1
        )
    
    # Hygiene metrics (row 2, col 2)
    if 'compression_ratio' in metrics:
        steps, values = zip(*metrics['compression_ratio'])
        fig.add_trace(
            go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                      name='Compression Ratio', line=dict(color='cyan', width=2)),
            row=2, col=2
        )
    
    # Faithfulness (row 3)
    if 'entailment_score' in metrics:
        steps, values = zip(*metrics['entailment_score'])
        fig.add_trace(
            go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                      name='Entailment Score', line=dict(color='blue', width=2)),
            row=3, col=1
        )
    
    if 'outlier_rate' in metrics:
        steps, values = zip(*metrics['outlier_rate'])
        fig.add_trace(
            go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                      name='Outlier Rate', line=dict(color='red', width=2)),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text=f'Checkpoint Evaluation Results - {model_name}',
        title_x=0.5,
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Checkpoint Step", row=3, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="BERTScore F1", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    
    # Save HTML
    html_file = os.path.join(output_dir, 'checkpoint_visualization.html')
    fig.write_html(html_file)
    print(f"Saved: {html_file}")


def log_to_wandb(metrics: Dict[str, List[Tuple[int, float]]], 
                 model_name: str, wandb_project: str, wandb_run_name: str):
    """Log metrics to WandB for comparison."""
    if not HAS_WANDB:
        print("Skipping WandB logging (wandb not available)")
        return
    
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        tags=["checkpoint-visualization", "post-hoc-analysis"],
        reinit=True
    )
    
    # Log all metrics
    for metric_name, data_points in metrics.items():
        for step, value in data_points:
            wandb.log({f"checkpoint/{metric_name}": value}, step=step)
    
    # Create summary table
    summary_data = []
    for metric_name, data_points in metrics.items():
        if data_points:
            steps, values = zip(*data_points)
            summary_data.append({
                "metric": metric_name,
                "min": min(values),
                "max": max(values),
                "final": values[-1],
                "checkpoints": len(data_points)
            })
    
    if summary_data:
        wandb.log({"summary": wandb.Table(dataframe=summary_data)})
    
    wandb.finish()
    print(f"Logged to WandB: {wandb_project}/{wandb_run_name}")


def visualize_checkpoints(model_dirs: List[str], output_dir: str,
                         wandb_project: Optional[str] = None,
                         wandb_run_name: Optional[str] = None,
                         compare_models: bool = False):
    """Main visualization function."""
    
    all_results = {}
    model_names = []
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir.rstrip('/'))
        model_names.append(model_name)
        
        print(f"\n{'='*70}")
        print(f"Loading results for: {model_name}")
        print(f"Directory: {model_dir}")
        print(f"{'='*70}")
        
        results = load_checkpoint_results(model_dir)
        if not results:
            print(f"Warning: No checkpoint results found for {model_name}")
            continue
        
        print(f"Loaded {len(results)} checkpoint results")
        all_results[model_name] = results
    
    if not all_results:
        print("Error: No checkpoint results found in any model directory")
        return
    
    # Create visualizations for each model
    for model_name, results in all_results.items():
        print(f"\n{'='*70}")
        print(f"Creating visualizations for: {model_name}")
        print(f"{'='*70}")
        
        metrics = extract_metrics(results)
        if not metrics:
            print(f"Warning: No metrics found for {model_name}")
            continue
        
        model_output_dir = os.path.join(output_dir, model_name) if compare_models else output_dir
        
        # Create matplotlib plots
        create_matplotlib_plots(metrics, model_output_dir, model_name)
        
        # Create Plotly HTML
        create_plotly_html(metrics, model_output_dir, model_name)
        
        # Log to WandB if requested
        if wandb_project:
            run_name = f"{model_name}-visualization" if compare_models else (wandb_run_name or f"{model_name}-visualization")
            log_to_wandb(metrics, model_name, wandb_project, run_name)
    
    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize checkpoint evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model_dir', type=str, nargs='+', required=True,
                       help='Model directory(ies) containing all_eval_results/ (can specify multiple for comparison)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualization files')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='WandB project name (optional, for logging to WandB)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='WandB run name (optional)')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare multiple models (creates separate directories for each)')
    
    args = parser.parse_args()
    
    visualize_checkpoints(
        model_dirs=args.model_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compare_models=args.compare_models
    )
