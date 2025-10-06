import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set color palette for contexts
CONTEXT_COLORS = {
    'score_context': {'leading': 'green', 'tied': 'blue', 'trailing': 'red'},
    'phase_context': {'early': 'gold', 'middle': 'dodgerblue', 'late': 'purple'},
    'intensity_context': {'low': 'gray', 'medium': 'orange', 'high': 'crimson'}
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_distribution_comparison(df, metrics, output_dir="results/plots/rq1"):
    """Box+violin plots for each metric by each context type (6x3 grid)"""
    ensure_dir(output_dir)
    fig, axes = plt.subplots(len(metrics), 3, figsize=(18, 3.5*len(metrics)), sharey='row')
    context_types = ['score_context', 'phase_context', 'intensity_context']
    for i, metric in enumerate(metrics):
        for j, context in enumerate(context_types):
            ax = axes[i, j]
            if context in df.columns and metric in df.columns:
                sns.violinplot(
                    x=context, y=metric, data=df, ax=ax,
                    palette=CONTEXT_COLORS[context], inner=None, alpha=0.7
                )
                sns.boxplot(
                    x=context, y=metric, data=df, ax=ax,
                    palette=CONTEXT_COLORS[context], width=0.2, boxprops=dict(alpha=0.5)
                )
                # Sample sizes
                counts = df[context].value_counts()
                for k, label in enumerate(counts.index):
                    ax.text(k, ax.get_ylim()[0], f"n={counts[label]}", ha='center', va='bottom', fontsize=9)
                ax.set_title(f"{metric.replace('_',' ').title()} by {context.replace('_',' ').title()}")
                ax.set_xlabel(context.replace('_',' ').title())
                ax.set_ylabel(metric.replace('_',' ').title())
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_distribution_comparison.png", dpi=300)
    plt.close()

def compute_effect_sizes(df, metrics, context_types):
    """Return DataFrame of Cohen's d for all pairwise context comparisons per metric."""
    effect_data = []
    for context in context_types:
        cats = df[context].dropna().unique()
        for metric in metrics:
            for a, b in combinations(cats, 2):
                group_a = df[df[context] == a][metric].dropna()
                group_b = df[df[context] == b][metric].dropna()
                if len(group_a) > 1 and len(group_b) > 1:
                    # Cohen's d
                    m1, m2 = group_a.mean(), group_b.mean()
                    s1, s2 = group_a.std(), group_b.std()
                    n1, n2 = len(group_a), len(group_b)
                    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                    d = (m1 - m2) / s_pooled if s_pooled > 0 else np.nan
                    effect_data.append({
                        'Metric': metric,
                        'Context': context,
                        'Group1': a,
                        'Group2': b,
                        'Cohens_d': d
                    })
    return pd.DataFrame(effect_data)

def plot_effect_size_heatmap(effect_df, output_dir="results/plots/rq1"):
    """Heatmap of effect sizes (Cohen's d) for all metrics and context pairs."""
    ensure_dir(output_dir)
    # Pivot to matrix: rows=metric, columns=context-pair
    effect_df['Pair'] = effect_df['Group1'] + ' vs ' + effect_df['Group2']
    pivot = effect_df.pivot_table(index='Metric', columns='Pair', values='Cohens_d')
    plt.figure(figsize=(2+pivot.shape[1], 1.5*pivot.shape[0]))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Figure 2: Effect Sizes (Cohen's d) for Contextual Comparisons")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_effect_size_heatmap.png", dpi=300)
    plt.close()

def plot_mean_trajectory(df, metrics, output_dir="results/plots/rq1"):
    """Line plots showing mean metric evolution across context categories."""
    ensure_dir(output_dir)
    context_types = ['score_context', 'phase_context', 'intensity_context']
    
    # Fix: Create a grid with enough subplots for all metric × context combinations
    fig, axes = plt.subplots(len(metrics), 3, figsize=(18, 3.5*len(metrics)))
    
    for i, metric in enumerate(metrics):
        for j, context in enumerate(context_types):
            ax = axes[i, j]  # Use 2D indexing instead of flattened
            if context in df.columns and metric in df.columns:
                means = df.groupby(context)[metric].mean()
                stds = df.groupby(context)[metric].std()
                cats = means.index
                ax.errorbar(
                    range(len(cats)), means, yerr=stds, fmt='-o',
                    color='black', ecolor='gray', capsize=4
                )
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats)
                ax.set_title(f"{metric.replace('_',' ').title()} by {context.replace('_',' ').title()}")
                ax.set_xlabel(context.replace('_',' ').title())
                ax.set_ylabel(metric.replace('_',' ').title())
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_mean_trajectory.png", dpi=300)
    plt.close()
    
def statistical_test_summary(df, metrics, context_types, output_dir="results/plots/rq1"):
    """Save a CSV summary of statistical tests (ANOVA/Tukey) for each metric/context."""
    ensure_dir(output_dir)
    summary = []
    for context in context_types:
        for metric in metrics:
            if context in df.columns and metric in df.columns:
                cats = df[context].dropna().unique()
                groups = [df[df[context] == cat][metric].dropna() for cat in cats]
                if len(groups) > 1:
                    # ANOVA
                    from scipy.stats import f_oneway
                    fval, pval = f_oneway(*groups)
                    summary.append({
                        'Metric': metric,
                        'Context': context,
                        'Test': 'ANOVA',
                        'F': fval,
                        'p': pval
                    })
                    # Tukey HSD for pairwise
                    tukey = pairwise_tukeyhsd(df[metric], df[context])
                    
                    for res in tukey.summary().data[1:]:
                        #print("Tukey row:", res, "Length:", len(res))
                        summary.append({
                            'Metric': metric,
                            'Context': context,
                            'Test': 'Tukey',
                            'Group1': res[0],
                            'Group2': res[1],
                            'meandiff': res[2],
                            'p-adj': res[4],
                            'lower': res[5],
                            'upper': res[6],
                            'reject': res[7] if len(res) > 7 else None
                        })
    pd.DataFrame(summary).to_csv(f"{output_dir}/statistical_test_summary.csv", index=False)

def plot_forest_pairwise(effect_df, output_dir="results/plots/rq1"):
    """Forest plot for significant pairwise comparisons."""
    ensure_dir(output_dir)
    sig = effect_df[effect_df['Cohens_d'].abs() > 0.2]  # threshold for practical significance
    if sig.empty:
        print("No significant pairwise effects found.")
        return
    plt.figure(figsize=(10, 0.5*len(sig)))
    y = np.arange(len(sig))
    plt.errorbar(sig['Cohens_d'], y, xerr=0.1, fmt='o', color='black')
    plt.yticks(y, sig['Metric'] + ": " + sig['Group1'] + " vs " + sig['Group2'])
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Cohen's d")
    plt.title("Figure 4: Forest Plot of Significant Pairwise Effects")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_forest_pairwise.png", dpi=300)
    plt.close()

def plot_context_cooccurrence(df, output_dir="results/plots/rq1"):
    """Heatmap of frequency of context combinations."""
    ensure_dir(output_dir)
    if all(c in df.columns for c in ['score_context', 'phase_context', 'intensity_context']):
        cooc = pd.crosstab(
            [df['score_context'], df['phase_context']],
            df['intensity_context']
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(cooc, annot=True, fmt='d', cmap='Blues')
        plt.title("Figure 5: Context Co-occurrence Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure5_context_cooccurrence.png", dpi=300)
        plt.close()

def plot_example_networks(network_data, output_dir="results/plots/rq1"):
    """3x3 grid of representative passing networks for leading/tied/trailing × early/middle/late."""
    import networkx as nx
    ensure_dir(output_dir)
    score_states = ['leading', 'tied', 'trailing']
    phases = ['early', 'middle', 'late']
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    for i, score in enumerate(score_states):
        for j, phase in enumerate(phases):
            ax = axes[i, j]
            # Find a representative network for this context
            for d in network_data:
                if d.get('score_context') == score and d.get('phase_context') == phase and d.get('network') is not None:
                    G = d['network']
                    break
            else:
                ax.axis('off')
                continue
            pos = {n: (n % 7, n // 7) for n in G.nodes}
            node_centrality = nx.betweenness_centrality(G)
            node_activity = np.array([G.degree(n, weight='weight') for n in G.nodes])
            node_colors = node_activity
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=[300 + 2000*node_centrality[n] for n in G.nodes],
                node_color=node_colors, cmap='YlOrRd', alpha=0.85
            )
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=[2*G[u][v].get('weight', 1) for u, v in G.edges],
                alpha=0.6
            )
            ax.set_title(f"{score.title()} / {phase.title()}")
            ax.set_xlim(-1, 7)
            ax.set_ylim(-1, 7)
            ax.axis('off')
    plt.suptitle("Figure 6: Example Passing Networks by Context", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure6_example_networks.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    from tactical_analysis_system.main_analysis import MainAnalysis
    from tactical_analysis_system.data_loader import DataLoader
    from tactical_analysis_system.visualizer import RQ1Visualizer
    import pandas as pd
    
    window_size = 10
    data_file = "statsbomb_data_interim_100.json"
    analysis = MainAnalysis(use_saved_data=True, data_file=data_file, window_size=window_size)

    print("Running RQ1 Analysis...")
    rq1_results = analysis.run_rq1_analysis(max_matches=10, save_results=True, filepath=data_file)
    print("RQ1 Done")

    import pandas as pd

    if 'network_metrics' in rq1_results:
        results_df = pd.DataFrame(rq1_results['network_metrics'])
    else:
        raise ValueError("No results DataFrame found in rq1_results")

    network_data = rq1_results.get('network_data', [])

    metrics = [
        'density', 'clustering_coefficient', 'avg_betweenness_centrality',
        'avg_eigenvector_centrality', 'avg_path_length', 'centralization'
    ]
    context_types = ['score_context', 'phase_context', 'intensity_context']

    # 1. Distribution comparison
    plot_distribution_comparison(results_df, metrics)

    # 2. Effect sizes
    effect_df = compute_effect_sizes(results_df, metrics, context_types)
    plot_effect_size_heatmap(effect_df)

    # 3. Mean trajectory
    plot_mean_trajectory(results_df, metrics)

    # 4. Statistical test summary (CSV)
    statistical_test_summary(results_df, metrics, context_types)

    # 5. Forest plot for pairwise comparisons
    plot_forest_pairwise(effect_df)

    # 6. Context co-occurrence heatmap
    plot_context_cooccurrence(results_df)

    # 7. Example networks (if you have network_data)
    if network_data:
        plot_example_networks(network_data)