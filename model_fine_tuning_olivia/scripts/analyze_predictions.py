"""
Analyze predictions from evaluation runs, focusing on repetition issues.

This script analyzes prediction files (e.g., checkpoint-4100-inputs-refs-preds.jsonl)
to identify and quantify repetition problems, especially for gemma models.

Usage:
    python analyze_predictions.py \
        --predictions_file models/gemma-2-9b-apptainer-fsdp/all_eval_results/regular-checkpoint-4100-inputs-refs-preds.jsonl \
        --rep_threshold 0.5 \
        --output_dir analysis_results
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
try:
    import statistics
except ImportError:
    # Fallback for Python < 3.4
    import math
    def quantiles(data, n=4):
        """Simple quantile calculation."""
        sorted_data = sorted(data)
        k = len(sorted_data)
        return [sorted_data[int(i * (k-1) / (n-1))] for i in range(n-1)]
    statistics.quantiles = quantiles

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


def remove_markup(text: str) -> str:
    """Remove markup tokens like ### and other formatting from text.
    
    This ensures markup doesn't affect metric calculations.
    """
    # Remove common markup patterns
    text = re.sub(r'###+', '', text)  # Remove ### markers
    text = re.sub(r'\*\*+', '', text)  # Remove ** bold markers
    text = re.sub(r'__+', '', text)    # Remove __ underline markers
    text = re.sub(r'~~+', '', text)    # Remove ~~ strikethrough markers
    # Remove any remaining standalone special characters used as markup
    text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
    return text.strip()


def ngram_repetition(doc: str, n: int = 3) -> float:
    """Calculate n-gram repetition score.
    
    Returns the ratio of repeated n-grams to total n-grams.
    Higher values indicate more repetition.
    """
    tokens = re.findall(r"\d+(?:[.,]\d+)?|[\w/-]+|[^\w\s]", doc.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    c = Counter(ngrams)
    total = sum(c.values())
    repeated = sum(v for v in c.values() if v > 1)
    return repeated / total if total else 0.0


def find_repetitive_sequences(text: str, min_repeat: int = 3) -> List[Tuple[str, int]]:
    """Find 3-gram sequences that repeat at least min_repeat times.
    
    Returns list of (sequence, count) tuples.
    """
    tokens = re.findall(r"\d+(?:[.,]\d+)?|[\w/-]+|[^\w\s]", text.lower())
    if len(tokens) < 3:
        return []
    
    # Only check for 3-gram repetitions
    ngrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    c = Counter(ngrams)
    repetitive_seqs = []
    for ngram, count in c.items():
        if count >= min_repeat:
            seq_str = " ".join(ngram)
            repetitive_seqs.append((seq_str, count))
    
    return repetitive_seqs


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "q25": statistics.quantiles(values, n=4)[0] if len(values) > 1 else values[0],
        "q75": statistics.quantiles(values, n=4)[2] if len(values) > 1 else values[0],
    }


def analyze_predictions(
    predictions_file: str,
    rep_threshold: float = 0.5,
    output_dir: str = None
) -> Dict:
    """Analyze predictions for repetition and other issues.
    
    Args:
        predictions_file: Path to JSONL file with predictions
        rep_threshold: Threshold for considering a prediction as having high repetition
        output_dir: Directory to save analysis results (optional)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Loading predictions from: {predictions_file}")
    
    # Load predictions
    predictions_data = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions_data.append(json.loads(line))
    
    total_predictions = len(predictions_data)
    print(f"Loaded {total_predictions} predictions")
    
    # Calculate metrics for each prediction
    analysis_results = []
    rep_3gram_scores = []
    pred_lengths = []
    ref_lengths = []
    compression_ratios = []
    
    high_rep_predictions = []
    empty_predictions = []
    
    # Track repetitive sequences
    all_repetitive_seqs = defaultdict(int)
    
    for idx, entry in enumerate(predictions_data):
        prediction_raw = entry.get("prediction", "").strip()
        reference = entry.get("reference", "").strip()
        input_text = entry.get("input_text", "").strip()
        
        # Remove markup before calculating metrics
        prediction = remove_markup(prediction_raw)
        
        # Calculate rep_3gram
        rep_3gram = ngram_repetition(prediction, n=3)
        rep_3gram_scores.append(rep_3gram)
        
        # Length statistics
        pred_words = len(re.findall(r"\w+", prediction))
        ref_words = len(re.findall(r"\w+", reference))
        input_words = len(re.findall(r"\w+", input_text))
        
        pred_lengths.append(pred_words)
        ref_lengths.append(ref_words)
        
        compression_ratio = (pred_words / input_words) if input_words > 0 else None
        if compression_ratio is not None:
            compression_ratios.append(compression_ratio)
        
        # Check for empty predictions (use cleaned prediction)
        if not prediction or pred_words == 0:
            empty_predictions.append({
                "index": idx,
                "prediction": prediction,  # Full prediction (already cleaned)
                "rep_3gram": rep_3gram
            })
        
        # Check for high repetition
        if rep_3gram >= rep_threshold:
            repetitive_seqs = find_repetitive_sequences(prediction, min_repeat=3)
            for seq, count in repetitive_seqs:
                all_repetitive_seqs[seq] += count
            
            high_rep_predictions.append({
                "index": idx,
                "rep_3gram": rep_3gram,
                "prediction_length": pred_words,
                "prediction": prediction,  # Always print full prediction
                "repetitive_sequences": repetitive_seqs[:5]  # Top 5
            })
        
        analysis_results.append({
            "index": idx,
            "rep_3gram": rep_3gram,
            "pred_length": pred_words,
            "ref_length": ref_words,
            "compression_ratio": compression_ratio,
            "has_high_rep": rep_3gram >= rep_threshold,
            "is_empty": pred_words == 0
        })
    
    # Calculate overall statistics
    rep_stats = calculate_statistics(rep_3gram_scores)
    pred_length_stats = calculate_statistics(pred_lengths)
    ref_length_stats = calculate_statistics(ref_lengths)
    compression_stats = calculate_statistics(compression_ratios) if compression_ratios else {}
    
    # Count predictions by category
    high_rep_count = len(high_rep_predictions)
    empty_count = len(empty_predictions)
    normal_count = total_predictions - high_rep_count - empty_count
    
    # Calculate percentage with high repetition
    high_rep_percentage = (high_rep_count / total_predictions * 100) if total_predictions > 0 else 0
    empty_percentage = (empty_count / total_predictions * 100) if total_predictions > 0 else 0
    
    # Distribution of rep_3gram scores
    rep_distribution = {
        "0.0-0.1": sum(1 for r in rep_3gram_scores if 0.0 <= r < 0.1),
        "0.1-0.3": sum(1 for r in rep_3gram_scores if 0.1 <= r < 0.3),
        "0.3-0.5": sum(1 for r in rep_3gram_scores if 0.3 <= r < 0.5),
        "0.5-0.7": sum(1 for r in rep_3gram_scores if 0.5 <= r < 0.7),
        "0.7-0.9": sum(1 for r in rep_3gram_scores if 0.7 <= r < 0.9),
        "0.9-1.0": sum(1 for r in rep_3gram_scores if 0.9 <= r <= 1.0),
    }
    
    # Most common repetitive sequences
    top_repetitive_seqs = sorted(all_repetitive_seqs.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Compare high-rep vs normal predictions
    high_rep_rep_scores = [r["rep_3gram"] for r in high_rep_predictions]
    normal_rep_scores = [r["rep_3gram"] for r in analysis_results if not r["has_high_rep"] and not r["is_empty"]]
    
    high_rep_rep_stats = calculate_statistics(high_rep_rep_scores) if high_rep_rep_scores else {}
    normal_rep_stats = calculate_statistics(normal_rep_scores) if normal_rep_scores else {}
    
    # Summary
    summary = {
        "file": predictions_file,
        "total_predictions": total_predictions,
        "threshold_used": rep_threshold,
        
        # Repetition statistics
        "repetition": {
            "overall_stats": rep_stats,
            "high_rep_count": high_rep_count,
            "high_rep_percentage": high_rep_percentage,
            "distribution": rep_distribution,
            "high_rep_stats": high_rep_rep_stats,
            "normal_stats": normal_rep_stats,
        },
        
        # Length statistics
        "lengths": {
            "prediction": pred_length_stats,
            "reference": ref_length_stats,
            "compression_ratio": compression_stats,
        },
        
        # Empty predictions
        "empty_predictions": {
            "count": empty_count,
            "percentage": empty_percentage,
        },
        
        # Categories
        "categories": {
            "high_rep": high_rep_count,
            "normal": normal_count,
            "empty": empty_count,
        },
        
        # Most repetitive sequences
        "top_repetitive_sequences": top_repetitive_seqs,
        
        # Sample problematic predictions
        "sample_high_rep": high_rep_predictions[:10],  # First 10
        "sample_empty": empty_predictions[:10],  # First 10
    }
    
    # Print summary
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total predictions: {total_predictions}")
    print(f"\nRepetition Analysis (threshold: {rep_threshold}):")
    print(f"  High repetition: {high_rep_count} ({high_rep_percentage:.1f}%)")
    print(f"  Normal: {normal_count} ({100-high_rep_percentage-empty_percentage:.1f}%)")
    print(f"  Empty: {empty_count} ({empty_percentage:.1f}%)")
    print(f"\nRep_3gram Statistics:")
    print(f"  Mean: {rep_stats['mean']:.4f}")
    print(f"  Median: {rep_stats['median']:.4f}")
    print(f"  Std Dev: {rep_stats['stdev']:.4f}")
    print(f"  Min: {rep_stats['min']:.4f}, Max: {rep_stats['max']:.4f}")
    print(f"\nRep_3gram Distribution:")
    for range_name, count in rep_distribution.items():
        percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
        print(f"  {range_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nLength Statistics:")
    print(f"  Prediction: mean={pred_length_stats['mean']:.1f}, median={pred_length_stats['median']:.1f}")
    print(f"  Reference: mean={ref_length_stats['mean']:.1f}, median={ref_length_stats['median']:.1f}")
    if compression_stats:
        print(f"  Compression ratio: mean={compression_stats['mean']:.3f}, median={compression_stats['median']:.3f}")
    
    if top_repetitive_seqs:
        print(f"\nTop 10 Most Common Repetitive Sequences:")
        for i, (seq, count) in enumerate(top_repetitive_seqs[:10], 1):
            print(f"  {i}. '{seq}' (appears {count} times)")
    
    if high_rep_predictions:
        print(f"\nSample High-Repetition Predictions (first 3):")
        for i, pred_info in enumerate(high_rep_predictions[:3], 1):
            print(f"\n  {i}. Index {pred_info['index']}, rep_3gram={pred_info['rep_3gram']:.3f}")
            print(f"     Length: {pred_info['prediction_length']} words")
            print(f"     Preview: {pred_info['prediction']}")
            if pred_info['repetitive_sequences']:
                print(f"     Repetitive sequences: {pred_info['repetitive_sequences'][:3]}")
    
    # Create visualizations
    if VISUALIZATION_AVAILABLE and output_dir:
        base_name = os.path.basename(predictions_file).replace(".jsonl", "")
        create_visualizations(
            rep_3gram_scores=rep_3gram_scores,
            pred_lengths=pred_lengths,
            ref_lengths=ref_lengths,
            compression_ratios=compression_ratios,
            high_rep_indices=[r["index"] for r in high_rep_predictions],
            rep_threshold=rep_threshold,
            top_repetitive_seqs=top_repetitive_seqs[:10],
            output_dir=output_dir,
            base_name=base_name
        )
    
    # Save detailed results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(predictions_file).replace(".jsonl", "")
        
        # Save summary
        summary_file = os.path.join(output_dir, f"{base_name}-analysis-summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_file}")
        
        # Save all high-rep predictions
        if high_rep_predictions:
            high_rep_file = os.path.join(output_dir, f"{base_name}-high-rep-predictions.json")
            with open(high_rep_file, 'w', encoding='utf-8') as f:
                json.dump(high_rep_predictions, f, indent=2, ensure_ascii=False)
            print(f"High-rep predictions saved to: {high_rep_file}")
        
        # Save all analysis results
        all_results_file = os.path.join(output_dir, f"{base_name}-all-analysis.json")
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"All analysis results saved to: {all_results_file}")
    
    return summary


def create_visualizations(
    rep_3gram_scores: List[float],
    pred_lengths: List[int],
    ref_lengths: List[int],
    compression_ratios: List[float],
    high_rep_indices: List[int],
    rep_threshold: float,
    top_repetitive_seqs: List[Tuple[str, int]],
    output_dir: str,
    base_name: str
):
    """Create visualization plots for the analysis."""
    if not VISUALIZATION_AVAILABLE:
        return
    
    print("\nGenerating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig_size = (12, 8)
    
    # 1. Histogram of rep_3gram scores
    fig, ax = plt.subplots(figsize=fig_size)
    ax.hist(rep_3gram_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(rep_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({rep_threshold})')
    ax.axvline(statistics.mean(rep_3gram_scores), color='green', linestyle='--', linewidth=2, label=f'Mean ({statistics.mean(rep_3gram_scores):.3f})')
    ax.set_xlabel('Rep_3gram Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Rep_3gram Scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}-rep3gram-histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {base_name}-rep3gram-histogram.png")
    
    # 2. Scatter plot: rep_3gram vs prediction length
    fig, ax = plt.subplots(figsize=fig_size)
    high_rep_mask = [i in high_rep_indices for i in range(len(rep_3gram_scores))]
    normal_mask = [not m for m in high_rep_mask]
    
    ax.scatter(
        [pred_lengths[i] for i in range(len(pred_lengths)) if normal_mask[i]],
        [rep_3gram_scores[i] for i in range(len(rep_3gram_scores)) if normal_mask[i]],
        alpha=0.5, s=20, color='blue', label='Normal predictions'
    )
    ax.scatter(
        [pred_lengths[i] for i in range(len(pred_lengths)) if high_rep_mask[i]],
        [rep_3gram_scores[i] for i in range(len(rep_3gram_scores)) if high_rep_mask[i]],
        alpha=0.7, s=30, color='red', label='High repetition', marker='x'
    )
    ax.axhline(rep_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Prediction Length (words)', fontsize=12)
    ax.set_ylabel('Rep_3gram Score', fontsize=12)
    ax.set_title('Rep_3gram vs Prediction Length', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}-rep3gram-vs-length.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {base_name}-rep3gram-vs-length.png")
    
    # 3. Box plot: High-rep vs Normal predictions
    if high_rep_indices:
        fig, ax = plt.subplots(figsize=(10, 6))
        high_rep_scores = [rep_3gram_scores[i] for i in high_rep_indices]
        normal_scores = [rep_3gram_scores[i] for i in range(len(rep_3gram_scores)) if i not in high_rep_indices]
        
        box_data = [normal_scores, high_rep_scores]
        box_labels = ['Normal', 'High Repetition']
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Rep_3gram Score', fontsize=12)
        ax.set_title('Rep_3gram Distribution: Normal vs High Repetition', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}-rep3gram-boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {base_name}-rep3gram-boxplot.png")
    
    # 4. Distribution of compression ratios
    if compression_ratios:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.hist(compression_ratios, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(statistics.mean(compression_ratios), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean ({statistics.mean(compression_ratios):.3f})')
        ax.set_xlabel('Compression Ratio (prediction/input)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Compression Ratios', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}-compression-ratio-histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {base_name}-compression-ratio-histogram.png")
    
    # 5. Prediction vs Reference length comparison
    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(ref_lengths, pred_lengths, alpha=0.5, s=20, color='purple')
    # Add diagonal line (y=x)
    max_len = max(max(pred_lengths) if pred_lengths else 0, max(ref_lengths) if ref_lengths else 0)
    ax.plot([0, max_len], [0, max_len], 'r--', linewidth=2, label='y=x (equal length)')
    ax.set_xlabel('Reference Length (words)', fontsize=12)
    ax.set_ylabel('Prediction Length (words)', fontsize=12)
    ax.set_title('Prediction vs Reference Length', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}-pred-vs-ref-length.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {base_name}-pred-vs-ref-length.png")
    
    # 6. Top repetitive sequences bar chart
    if top_repetitive_seqs:
        fig, ax = plt.subplots(figsize=(12, 8))
        sequences = [seq[:50] + '...' if len(seq) > 50 else seq for seq, _ in top_repetitive_seqs]
        counts = [count for _, count in top_repetitive_seqs]
        
        y_pos = np.arange(len(sequences))
        bars = ax.barh(y_pos, counts, color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sequences, fontsize=9)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title('Top 10 Most Common Repetitive Sequences', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}-top-repetitive-sequences.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {base_name}-top-repetitive-sequences.png")
    
    # 7. Cumulative distribution of rep_3gram
    fig, ax = plt.subplots(figsize=fig_size)
    sorted_scores = sorted(rep_3gram_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cumulative, linewidth=2, color='darkblue')
    ax.axvline(rep_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({rep_threshold})')
    ax.set_xlabel('Rep_3gram Score', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution of Rep_3gram Scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}-rep3gram-cumulative.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {base_name}-rep3gram-cumulative.png")
    
    print("  All visualizations generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze predictions for repetition and other issues"
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to JSONL file with predictions (e.g., checkpoint-4100-inputs-refs-preds.jsonl)"
    )
    parser.add_argument(
        "--rep_threshold",
        type=float,
        default=0.5,
        help="Threshold for considering a prediction as having high repetition (default: 0.5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: same directory as predictions file)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions_file) or "."
    
    # Run analysis
    analyze_predictions(
        predictions_file=args.predictions_file,
        rep_threshold=args.rep_threshold,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
