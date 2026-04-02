"""
Visualize checkpoint evaluation results from all_eval_results/ directory.

This script loads all checkpoint evaluation JSON files and creates comprehensive
visualizations showing training progress across metrics.

Usage:
    # Basic visualization (outputs to ./visualisations/<model_name>/)
    python visualise_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp

    # Custom output directory
    python visualise_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp \
        --output_dir visualizations/gemma-2-9b
    
    # Compare multiple models
    python visualise_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp models/gemma-7b-it-apptainer-fsdp \
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
    match = re.search(r'checkpoint-(\d+)-eval-results', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint_results(model_dir: str) -> Dict[int, Dict]:
    """Load 500-example checkpoint evaluation results from all_eval_results/.
    
    Returns:
        Dictionary mapping checkpoint step -> evaluation results
    """
    # Canonical 500-example filenames: checkpoint-N-eval-results-500-examples.json
    results = _load_results_by_pattern(model_dir, "checkpoint-*-eval-results-500-examples.json")
    if results:
        return results
    # Backward compatibility: unsuffixed files were commonly 500-example outputs.
    return _load_results_by_pattern(model_dir, "checkpoint-*-eval-results.json",
                                    exclude_pattern="examples_")


def load_checkpoint_results_1000(model_dir: str) -> Dict[int, Dict]:
    """Load 1000-example checkpoint evaluation results from all_eval_results/.
    
    Returns:
        Dictionary mapping checkpoint step -> evaluation results
    """
    # Canonical 1000-example filenames: checkpoint-N-eval-results-1000-examples.json
    results = _load_results_by_pattern(model_dir, "checkpoint-*-eval-results-1000-examples.json")
    if results:
        return results
    # Backward compatibility with legacy suffix.
    return _load_results_by_pattern(model_dir, "checkpoint-*-eval-results-examples_1000.json")


def _load_results_by_pattern(model_dir: str, file_pattern: str,
                             exclude_pattern: Optional[str] = None) -> Dict[int, Dict]:
    all_eval_results_dir = os.path.join(model_dir, "all_eval_results")
    if not os.path.exists(all_eval_results_dir):
        print(f"Warning: all_eval_results directory not found: {all_eval_results_dir}")
        return {}
    
    results = {}
    pattern = os.path.join(all_eval_results_dir, file_pattern)
    
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        if exclude_pattern and exclude_pattern in basename:
            continue
        try:
            step = extract_checkpoint_step(basename)
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
        
        # Extended metrics - Faithfulness / NLI
        # Primary: eval_faithfulness (saved by evaluate_distributed_checkpoints_multigpu).
        # Alternate: top-level "faithfulness" (same inner keys; some pipelines omit the eval_ prefix).
        # Outliers: mean_ratio_outliers (aggregate API) or mean_outlier_rate (older / subset exports).
        faith_block = None
        if isinstance(data.get("eval_faithfulness"), dict):
            faith_block = data["eval_faithfulness"]
        elif isinstance(data.get("faithfulness"), dict):
            faith_block = data["faithfulness"]

        got_entailment = False
        got_outlier = False
        if faith_block:
            if "mean_entailment_score" in faith_block:
                if "entailment_score" not in metrics:
                    metrics["entailment_score"] = []
                metrics["entailment_score"].append((step, faith_block["mean_entailment_score"]))
                got_entailment = True
            out_val = faith_block.get("mean_ratio_outliers")
            if out_val is None:
                out_val = faith_block.get("mean_outlier_rate")
            if out_val is not None:
                if "outlier_rate" not in metrics:
                    metrics["outlier_rate"] = []
                metrics["outlier_rate"].append((step, out_val))
                got_outlier = True

        if not got_entailment and "eval_faithfulness_mean_entailment_score" in data:
            if "entailment_score" not in metrics:
                metrics["entailment_score"] = []
            metrics["entailment_score"].append((step, data["eval_faithfulness_mean_entailment_score"]))

        if not got_outlier and "eval_faithfulness_mean_ratio_outliers" in data:
            if "outlier_rate" not in metrics:
                metrics["outlier_rate"] = []
            metrics["outlier_rate"].append((step, data["eval_faithfulness_mean_ratio_outliers"]))
        elif not got_outlier and "eval_faithfulness_mean_outlier_rate" in data:
            if "outlier_rate" not in metrics:
                metrics["outlier_rate"] = []
            metrics["outlier_rate"].append((step, data["eval_faithfulness_mean_outlier_rate"]))
    
    return metrics


def _plot_metric_dual(ax, metric_name: str, metrics_500: Dict, metrics_1000: Dict,
                      color: str = None, label: str = None, ylim_bottom=None, ylim_top=None,
                      **plot_kwargs):
    """Plot a single metric with both 500-example (faded) and 1000-example (emphasised) data.
    
    500-example data is drawn first as a faded background trace.
    1000-example data is drawn on top with full emphasis.
    """
    has_500 = metric_name in metrics_500
    has_1000 = metric_name in metrics_1000

    if has_500:
        steps, values = zip(*metrics_500[metric_name])
        ax.plot(steps, values, marker='o', linewidth=1.2, markersize=4, alpha=0.30,
                color=color, label=f'{label} (500-ex)' if label else None,
                **plot_kwargs)
    if has_1000:
        steps, values = zip(*metrics_1000[metric_name])
        ax.plot(steps, values, marker='o', linewidth=2.5, markersize=6, alpha=1.0,
                color=color, label=f'{label} (1000-ex)' if label else None,
                **plot_kwargs)

    if ylim_bottom is not None or ylim_top is not None:
        current_bottom, current_top = ax.get_ylim()
        ax.set_ylim(
            bottom=ylim_bottom if ylim_bottom is not None else current_bottom,
            top=ylim_top if ylim_top is not None else current_top,
        )

    return has_500 or has_1000


def create_matplotlib_plots(metrics_500: Dict[str, List[Tuple[int, float]]],
                           output_dir: str, model_name: str,
                           metrics_1000: Optional[Dict[str, List[Tuple[int, float]]]] = None):
    """Create matplotlib plots for all metrics.
    
    When metrics_1000 is provided, both are overlaid: 500-example data as faded
    background, 1000-example data emphasised on top.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping matplotlib plots (matplotlib not available)")
        return
    
    if metrics_1000 is None:
        metrics_1000 = {}

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    has_dual = bool(metrics_1000)
    legend_note = " (faded = 500-ex, solid = 1000-ex)" if has_dual else ""
    
    # ROUGE metrics
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    rouge_colors = {'rouge1': '#1f77b4', 'rouge2': '#2ca02c', 'rougeL': '#ff7f0e', 'rougeLsum': '#d62728'}
    if any(m in metrics_500 or m in metrics_1000 for m in rouge_metrics):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'ROUGE Metrics - {model_name}{legend_note}',
                     fontsize=16, fontweight='bold')
        
        for idx, rouge_type in enumerate(rouge_metrics):
            ax = axes[idx // 2, idx % 2]
            has_data = _plot_metric_dual(ax, rouge_type, metrics_500, metrics_1000,
                                         color=rouge_colors[rouge_type], ylim_bottom=0)
            ax.set_title(f'ROUGE-{rouge_type.upper()}', fontweight='bold')
            ax.set_xlabel('Checkpoint Step')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            if not has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rouge_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/rouge_metrics.png")
    
    # Extended metrics - Reference-based
    if 'bertscore_f1' in metrics_500 or 'bertscore_f1' in metrics_1000:
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_metric_dual(ax, 'bertscore_f1', metrics_500, metrics_1000,
                          color='green', ylim_bottom=0, ylim_top=1)
        ax.set_title(f'BERTScore F1 - {model_name}{legend_note}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint Step')
        ax.set_ylabel('BERTScore F1')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bertscore.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/bertscore.png")
    
    # Extended metrics - Hygiene
    hygiene_metrics = ['compression_ratio', 'repetition_3gram', 'ends_with_punct']
    hygiene_colors = {'compression_ratio': '#1f77b4', 'repetition_3gram': '#ff7f0e', 'ends_with_punct': '#2ca02c'}
    if any(m in metrics_500 or m in metrics_1000 for m in hygiene_metrics):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Hygiene Metrics - {model_name}{legend_note}',
                     fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(hygiene_metrics):
            ax = axes[idx]
            has_data = _plot_metric_dual(
                ax, metric, metrics_500, metrics_1000,
                color=hygiene_colors[metric],
                ylim_bottom=0 if metric == 'ends_with_punct' else None,
                ylim_top=1 if metric == 'ends_with_punct' else None,
            )
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Checkpoint Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            if not has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hygiene_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/hygiene_metrics.png")
    
    # Extended metrics - Faithfulness
    faith_present = any(m in metrics_500 or m in metrics_1000
                        for m in ['entailment_score', 'outlier_rate'])
    if faith_present:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Faithfulness Metrics - {model_name}{legend_note}',
                     fontsize=16, fontweight='bold')
        
        if 'entailment_score' in metrics_500 or 'entailment_score' in metrics_1000:
            _plot_metric_dual(axes[0], 'entailment_score', metrics_500, metrics_1000,
                              color='blue', ylim_bottom=0, ylim_top=1)
        else:
            axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Mean Entailment Score', fontweight='bold')
        axes[0].set_xlabel('Checkpoint Step')
        axes[0].set_ylabel('Entailment Score')
        axes[0].grid(True, alpha=0.3)
        
        if 'outlier_rate' in metrics_500 or 'outlier_rate' in metrics_1000:
            _plot_metric_dual(axes[1], 'outlier_rate', metrics_500, metrics_1000,
                              color='red', ylim_bottom=0, ylim_top=1)
        else:
            axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Mean Outlier Rate', fontweight='bold')
        axes[1].set_xlabel('Checkpoint Step')
        axes[1].set_ylabel('Outlier Rate')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'faithfulness_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/faithfulness_metrics.png")
    
    # Combined overview plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # ROUGE-Lsum (primary metric)
    if 'rougeLsum' in metrics_500:
        steps, values = zip(*metrics_500['rougeLsum'])
        ax.plot(steps, values, marker='o', linewidth=1.5, markersize=5, alpha=0.30,
                label='ROUGE-Lsum (500-ex)', color='blue')
    if 'rougeLsum' in metrics_1000:
        steps, values = zip(*metrics_1000['rougeLsum'])
        ax.plot(steps, values, marker='o', linewidth=3, markersize=8, alpha=1.0,
                label='ROUGE-Lsum (1000-ex)', color='blue')

    # BERTScore if available
    if 'bertscore_f1' in metrics_500:
        steps, values = zip(*metrics_500['bertscore_f1'])
        values = [v * 100 for v in values]
        ax.plot(steps, values, marker='s', linewidth=1, markersize=4, alpha=0.30,
                label='BERTScore F1 ×100 (500-ex)', color='green', linestyle='--')
    if 'bertscore_f1' in metrics_1000:
        steps, values = zip(*metrics_1000['bertscore_f1'])
        values = [v * 100 for v in values]
        ax.plot(steps, values, marker='s', linewidth=2, markersize=6, alpha=1.0,
                label='BERTScore F1 ×100 (1000-ex)', color='green', linestyle='--')
    
    ax.set_title(f'Training Progress Overview - {model_name}{legend_note}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/overview.png")


def _add_plotly_dual_trace(fig, metric_name: str, display_name: str, color: str,
                           metrics_500: Dict, metrics_1000: Dict,
                           row: int, col: int):
    """Add both 500-example (faded) and 1000-example (solid) traces to a Plotly figure."""
    if metric_name in metrics_500:
        steps, values = zip(*metrics_500[metric_name])
        fig.add_trace(
            go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                      name=f'{display_name} (500-ex)',
                      line=dict(color=color, width=1),
                      marker=dict(size=4),
                      opacity=0.25,
                      legendgroup=display_name,
                      showlegend=True),
            row=row, col=col
        )
    if metric_name in metrics_1000:
        steps, values = zip(*metrics_1000[metric_name])
        fig.add_trace(
            go.Scatter(x=list(steps), y=list(values), mode='lines+markers',
                      name=f'{display_name} (1000-ex)',
                      line=dict(color=color, width=2.5),
                      marker=dict(size=7),
                      opacity=1.0,
                      legendgroup=display_name,
                      showlegend=True),
            row=row, col=col
        )


def create_plotly_html(metrics_500: Dict[str, List[Tuple[int, float]]],
                       output_dir: str, model_name: str,
                       metrics_1000: Optional[Dict[str, List[Tuple[int, float]]]] = None):
    """Create interactive Plotly HTML visualization.
    
    When metrics_1000 is provided, both are overlaid: 500-example as faded traces,
    1000-example as emphasised traces.
    """
    if not HAS_PLOTLY:
        print("Skipping Plotly HTML (plotly not available)")
        return
    
    if metrics_1000 is None:
        metrics_1000 = {}

    os.makedirs(output_dir, exist_ok=True)
    
    has_dual = bool(metrics_1000)
    subtitle = " (faded = 500-ex, solid = 1000-ex)" if has_dual else ""
    
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
        _add_plotly_dual_trace(fig, rouge_type, f'ROUGE-{rouge_type.upper()}',
                               rouge_colors[rouge_type], metrics_500, metrics_1000,
                               row=1, col=1)
    
    # BERTScore (row 2, col 1)
    _add_plotly_dual_trace(fig, 'bertscore_f1', 'BERTScore F1', 'purple',
                           metrics_500, metrics_1000, row=2, col=1)
    
    # Hygiene metrics (row 2, col 2)
    _add_plotly_dual_trace(fig, 'compression_ratio', 'Compression Ratio', 'cyan',
                           metrics_500, metrics_1000, row=2, col=2)
    
    # Faithfulness (row 3)
    _add_plotly_dual_trace(fig, 'entailment_score', 'Entailment Score', 'blue',
                           metrics_500, metrics_1000, row=3, col=1)
    _add_plotly_dual_trace(fig, 'outlier_rate', 'Outlier Rate', 'red',
                           metrics_500, metrics_1000, row=3, col=1)
    
    fig.update_layout(
        title_text=f'Checkpoint Evaluation Results - {model_name}{subtitle}',
        title_x=0.5,
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Checkpoint Step", row=3, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="BERTScore F1", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    
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


def load_all_model_data(model_dirs: List[str]) -> Tuple[Dict[str, Dict[int, Dict]], Dict[str, Dict[int, Dict]]]:
    """Load 500-example and 1000-example results for all model directories.
    
    Returns:
        (all_results_500, all_results_1000) dicts mapping model_name -> {step: results}
    """
    all_results_500 = {}
    all_results_1000 = {}

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir.rstrip('/'))

        print(f"\n{'='*70}")
        print(f"Loading results for: {model_name}")
        print(f"Directory: {model_dir}")
        print(f"{'='*70}")

        results_500 = load_checkpoint_results(model_dir)
        results_1000 = load_checkpoint_results_1000(model_dir)

        if not results_500 and not results_1000:
            print(f"Warning: No checkpoint results found for {model_name}")
            continue

        print(f"Loaded {len(results_500)} checkpoint results (500-example)")
        print(f"Loaded {len(results_1000)} checkpoint results (1000-example)")

        all_results_500[model_name] = results_500
        all_results_1000[model_name] = results_1000

    return all_results_500, all_results_1000


def default_output_dir(model_dirs: List[str], compare_models: bool) -> str:
    """Derive a default output directory under ./visualisations/."""
    if compare_models or len(model_dirs) > 1:
        return os.path.join("visualisations", "comparison")
    model_name = os.path.basename(model_dirs[0].rstrip('/'))
    return os.path.join("visualisations", model_name)


def visualize_checkpoints(model_dirs: List[str], output_dir: Optional[str] = None,
                         wandb_project: Optional[str] = None,
                         wandb_run_name: Optional[str] = None,
                         compare_models: bool = False):
    """Main visualization function.
    
    Loads both 500-example and 1000-example evaluation results and overlays
    them in the same plots (500-example faded, 1000-example emphasised).
    """

    if output_dir is None:
        output_dir = default_output_dir(model_dirs, compare_models)

    all_results_500, all_results_1000 = load_all_model_data(model_dirs)
    
    if not all_results_500 and not all_results_1000:
        print("Error: No checkpoint results found in any model directory")
        return
    
    all_model_names = set(all_results_500.keys()) | set(all_results_1000.keys())

    for model_name in sorted(all_model_names):
        print(f"\n{'='*70}")
        print(f"Creating visualizations for: {model_name}")
        print(f"{'='*70}")

        results_500 = all_results_500.get(model_name, {})
        results_1000 = all_results_1000.get(model_name, {})
        
        metrics_500 = extract_metrics(results_500) if results_500 else {}
        metrics_1000 = extract_metrics(results_1000) if results_1000 else {}

        if not metrics_500 and not metrics_1000:
            print(f"Warning: No metrics found for {model_name}")
            continue
        
        model_output_dir = os.path.join(output_dir, model_name) if compare_models else output_dir
        
        create_matplotlib_plots(metrics_500, model_output_dir, model_name,
                                metrics_1000=metrics_1000)
        create_plotly_html(metrics_500, model_output_dir, model_name,
                           metrics_1000=metrics_1000)
        
        if wandb_project:
            run_name = (f"{model_name}-visualization" if compare_models
                        else (wandb_run_name or f"{model_name}-visualization"))
            combined_metrics = {}
            for key in set(metrics_500.keys()) | set(metrics_1000.keys()):
                combined_metrics[key] = metrics_1000.get(key, metrics_500.get(key, []))
            log_to_wandb(combined_metrics, model_name, wandb_project, run_name)
    
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
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualization files (default: ./visualisations/<model_name>)')
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
