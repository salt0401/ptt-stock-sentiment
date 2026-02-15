"""Visualization plots for v1 vs v2 sentiment pipeline comparison.

Generates publication-ready figures saved to output/comparison/figures/.

Plots:
1. Dictionary score distributions (v1 vs v2 propagated words)
2. Push-tag mean scores by class (grouped bar chart)
3. Push-tag confusion matrices (side-by-side heatmaps)
4. FF5 regression comparison (coefficient + significance bar chart)
5. Daily sentiment time series (v2)
6. Summary metrics comparison (radar/bar chart)
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR

# Try to use a font that supports Chinese characters
_CN_FONTS = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
             'Noto Sans CJK TC', 'Arial Unicode MS']
_font_found = False
for _f in _CN_FONTS:
    if any(_f.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_f] + plt.rcParams['font.sans-serif']
        _font_found = True
        break
plt.rcParams['axes.unicode_minus'] = False

# Output
FIGURES_DIR = OUTPUT_DIR / "comparison" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
C_V1 = '#5B8DBE'   # Steel blue for v1
C_V2 = '#E07B54'   # Warm orange for v2
C_BASE = '#8C8C8C'  # Gray for baseline
PALETTE = [C_V1, C_V2]


def plot_1_distribution():
    """Plot 1: Dictionary score distributions — v1 vs v2 propagated words."""
    from column_map import SENTIMENT_COLUMNS

    v1 = pd.read_csv(OUTPUT_DIR / "computed_sentiment_dict.csv")
    v1.rename(columns=SENTIMENT_COLUMNS, inplace=True)
    v2 = pd.read_csv(OUTPUT_DIR / "v2" / "computed_sentiment_dict_v2.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlaid KDE
    ax = axes[0]
    v1_scores = v1['sentiment_score'].values
    v2_scores = v2['sentiment_score'].values

    ax.hist(v1_scores, bins=200, density=True, alpha=0.5, color=C_V1,
            label=f'v1 (n={len(v1):,})', range=(-0.5, 0.5))
    ax.hist(v2_scores, bins=200, density=True, alpha=0.5, color=C_V2,
            label=f'v2 (n={len(v2):,})', range=(-0.5, 0.5))
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Density')
    ax.set_title('Propagated Word Score Distributions')
    ax.legend()
    ax.set_xlim(-0.5, 0.5)

    # Right: zoomed on v1 to show its collapsed distribution
    ax2 = axes[1]
    ax2.hist(v1_scores, bins=200, density=True, alpha=0.7, color=C_V1,
             label='v1 (zoomed)')
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Density')
    ax2.set_title(f'v1 Propagated Scores (std={v1_scores.std():.4f})')
    ax2.axvline(v1_scores.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'mean={v1_scores.mean():.4f}')
    ax2.legend()

    fig.tight_layout()
    path = FIGURES_DIR / "1_score_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_2_pushtag_mean_scores():
    """Plot 2: Mean sentiment score by push-tag class."""
    # Full-dataset results (from the last comparison run)
    data = {
        'Class': ['Push', 'Boo', 'Arrow'],
        'v1': [0.0460, 0.0200, 0.0439],
        'v2': [-0.0082, -0.0201, -0.0049],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w/2, df['v1'], w, label='v1 (old)', color=C_V1, edgecolor='white')
    bars2 = ax.bar(x + w/2, df['v2'], w, label='v2 (new)', color=C_V2, edgecolor='white')

    ax.set_xlabel('Push Tag Class')
    ax.set_ylabel('Mean Sentiment Score')
    ax.set_title('Mean Sentiment Score by Push-Tag Class')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Class'])
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                f'{h:+.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        offset = 0.001 if h >= 0 else -0.003
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f'{h:+.4f}', ha='center', va=va, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "2_pushtag_mean_scores.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_3_confusion_matrices():
    """Plot 3: Confusion matrices for v1 and v2 (side by side)."""
    # Full-dataset results
    cm_v1 = np.array([
        [18291, 25741, 23893],
        [86675, 134648, 155098],
        [105060, 172945, 192444],
    ])
    cm_v2 = np.array([
        [42571, 49926, 25656],
        [188652, 270856, 159088],
        [277046, 368345, 212944],
    ])

    labels = ['Boo', 'Arrow', 'Push']

    # Normalize by row (true class)
    cm_v1_norm = cm_v1.astype(float) / cm_v1.sum(axis=1, keepdims=True)
    cm_v2_norm = cm_v2.astype(float) / cm_v2.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm_v1_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax1,
                vmin=0, vmax=0.5, cbar_kws={'label': 'Proportion'})
    ax1.set_title(f'v1 (old) — Coverage: 54.0%\nAccuracy: 0.3776, F1: 0.3227')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(cm_v2_norm, annot=True, fmt='.2%', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=ax2,
                vmin=0, vmax=0.5, cbar_kws={'label': 'Proportion'})
    ax2.set_title(f'v2 (new) — Coverage: 94.1%\nAccuracy: 0.3300, F1: 0.2964')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    fig.suptitle('Confusion Matrices (Row-Normalized)', fontsize=14, y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / "3_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_4_regression_comparison():
    """Plot 4: FF5 regression comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 4a: Delta R-squared
    ax = axes[0]
    vals = [0.000022, 0.000072]
    bars = ax.bar(['v1', 'v2'], vals, color=PALETTE, edgecolor='white', width=0.5)
    ax.set_title('Incremental R-squared\n(from adding SENT)')
    ax.set_ylabel(r'$\Delta R^2$')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.000002,
                f'{v:.6f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 0.0001)

    # 4b: t-statistic
    ax = axes[1]
    vals = [-1.6605, 2.3851]
    colors = [C_V1, C_V2]
    bars = ax.bar(['v1', 'v2'], vals, color=colors, edgecolor='white', width=0.5)
    ax.set_title('SENT t-statistic\n(HAC/Newey-West)')
    ax.set_ylabel('t-statistic')
    ax.axhline(1.96, color='green', linestyle='--', linewidth=1, alpha=0.7,
               label='5% critical value')
    ax.axhline(-1.96, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        offset = 0.1 if v >= 0 else -0.25
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=11)

    # 4c: Delta AIC
    ax = axes[2]
    vals = [-0.63, -6.77]
    bars = ax.bar(['v1', 'v2'], vals, color=PALETTE, edgecolor='white', width=0.5)
    ax.set_title('Delta AIC\n(negative = better fit)')
    ax.set_ylabel(r'$\Delta$ AIC')
    ax.axhline(0, color='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v - 0.3,
                f'{v:.2f}', ha='center', va='top', fontsize=11)
    ax.set_ylim(-9, 1)

    fig.suptitle('FF5 Regression: excess_return ~ SMB + HML + RMW + CMA [+ SENT]',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / "4_regression_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_5_daily_sentiment():
    """Plot 5: Daily sentiment time series (v2) with market return overlay."""
    daily = pd.read_csv(OUTPUT_DIR / "v2" / "daily_sentiment_v2.csv")
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date')

    # Load total_table for market returns
    total = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "financial" /
        "aggregated_table" / "total_table.csv"
    )
    total['Date'] = pd.to_datetime(total['Date'])

    # Merge
    merged = daily.merge(total[['Date', 'spread']], on='Date', how='inner')

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Sentiment (left axis)
    color_sent = C_V2
    ax1.plot(merged['Date'], merged['sentiment_mean'], color=color_sent,
             alpha=0.6, linewidth=0.8, label='V2 Daily Sentiment')
    # Rolling mean
    rolling = merged['sentiment_mean'].rolling(20, min_periods=5).mean()
    ax1.plot(merged['Date'], rolling, color=color_sent, linewidth=2,
             label='20-day Moving Avg')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Sentiment Score', color=color_sent)
    ax1.tick_params(axis='y', labelcolor=color_sent)
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # Market return (right axis)
    ax2 = ax1.twinx()
    color_ret = C_V1
    ax2.plot(merged['Date'], merged['spread'], color=color_ret,
             alpha=0.3, linewidth=0.5, label='Excess Return')
    rolling_ret = merged['spread'].rolling(20, min_periods=5).mean()
    ax2.plot(merged['Date'], rolling_ret, color=color_ret, linewidth=1.5,
             linestyle='--', label='20-day MA Return')
    ax2.set_ylabel('Excess Return (spread)', color=color_ret)
    ax2.tick_params(axis='y', labelcolor=color_ret)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.set_title('V2 Daily Sentiment vs Market Excess Return')
    fig.tight_layout()
    path = FIGURES_DIR / "5_daily_sentiment_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_6_summary_metrics():
    """Plot 6: Summary comparison — key metrics side by side."""
    metrics = {
        'Coverage': (0.540, 0.941),
        'Cohen\'s Kappa': (0.0028, 0.0069),
        'SENT p-value\n(lower=better)': (0.0968, 0.0171),
        'Delta R-sq\n(x10000)': (0.22, 0.72),
        'Delta AIC\n(magnitude)': (0.63, 6.77),
    }

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    for ax, (name, (v1, v2)) in zip(axes, metrics.items()):
        bars = ax.bar(['v1', 'v2'], [v1, v2], color=PALETTE, edgecolor='white', width=0.5)
        ax.set_title(name, fontsize=10)

        # Add value labels
        for bar, v in zip(bars, [v1, v2]):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(v1, v2) * 0.03,
                    f'{v}', ha='center', va='bottom', fontsize=9)

        # Highlight winner
        if 'lower' in name:
            winner = 0 if v1 < v2 else 1
        else:
            winner = 0 if v1 > v2 else 1
        bars[winner].set_edgecolor('green')
        bars[winner].set_linewidth(2)

    fig.suptitle('V1 vs V2: Key Performance Metrics', fontsize=14, y=1.05)
    fig.tight_layout()
    path = FIGURES_DIR / "6_summary_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_7_score_boxplot():
    """Plot 7: Box plot of comment-level scores by tag class for v1 vs v2."""
    # We'll generate this from sampled data
    from column_map import SENTIMENT_COLUMNS

    # Load dictionaries
    v1_dict = pd.read_csv(OUTPUT_DIR / "computed_sentiment_dict.csv")
    v1_dict.rename(columns=SENTIMENT_COLUMNS, inplace=True)
    from sentiment import load_opinion_dictionary
    from config import OPINION_DICT_PATH
    opinion = load_opinion_dictionary(OPINION_DICT_PATH)
    v1_full = pd.concat([v1_dict, opinion[['word', 'sentiment_score']]], ignore_index=True)
    v2_full = pd.read_csv(OUTPUT_DIR / "v2" / "total_dictionary_v2.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # V1 distribution
    ax1.hist(v1_dict['sentiment_score'], bins=100, color=C_V1, alpha=0.7,
             edgecolor='white', density=True)
    ax1.set_title(f'v1 Propagated Scores\nmean={v1_dict["sentiment_score"].mean():.4f}, '
                  f'std={v1_dict["sentiment_score"].std():.4f}')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Density')
    ax1.set_xlim(-0.4, 0.4)

    # V2 distribution
    v2_computed = pd.read_csv(OUTPUT_DIR / "v2" / "computed_sentiment_dict_v2.csv")
    ax2.hist(v2_computed['sentiment_score'], bins=100, color=C_V2, alpha=0.7,
             edgecolor='white', density=True)
    ax2.set_title(f'v2 Propagated Scores\nmean={v2_computed["sentiment_score"].mean():.4f}, '
                  f'std={v2_computed["sentiment_score"].std():.4f}')
    ax2.set_xlabel('Sentiment Score')
    ax2.set_xlim(-0.4, 0.4)

    fig.suptitle('Propagated Word Score Distributions', fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / "7_propagated_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("Generating comparison visualizations...")
    print(f"Output directory: {FIGURES_DIR}\n")

    print("[1/7] Dictionary score distributions...")
    plot_1_distribution()

    print("[2/7] Push-tag mean scores by class...")
    plot_2_pushtag_mean_scores()

    print("[3/7] Confusion matrices...")
    plot_3_confusion_matrices()

    print("[4/7] FF5 regression comparison...")
    plot_4_regression_comparison()

    print("[5/7] Daily sentiment time series...")
    plot_5_daily_sentiment()

    print("[6/7] Summary metrics comparison...")
    plot_6_summary_metrics()

    print("[7/7] Propagated score distributions...")
    plot_7_score_boxplot()

    print(f"\nDone! All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
