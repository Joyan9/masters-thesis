import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class RQ1Visualizer:

    def plot_winning_vs_losing_networks(self, network_data, context_col='score_context', save_plot=True, output_filename='winning_vs_losing_networks.png'):
        """
        Create a side-by-side visualization of passing networks for winning vs. losing contexts.
        Args:
            network_data: List of dicts, each with keys including 'network' (nx.Graph), 'score_context', etc.
            context_col: The context column to split on (default: 'score_context').
            save_plot: Whether to save the plot to file.
            output_filename: Name of the output file.
        """
        import networkx as nx
        # Filter for winning and losing contexts
        win_networks = [d['network'] for d in network_data if d.get(context_col) == 'leading' and d['network'] is not None]
        lose_networks = [d['network'] for d in network_data if d.get(context_col) == 'trailing' and d['network'] is not None]

        # Aggregate networks (sum edge weights)
        def aggregate_networks(networks):
            G_agg = nx.Graph()
            for G in networks:
                for u, v, d in G.edges(data=True):
                    w = d.get('weight', 1)
                    if G_agg.has_edge(u, v):
                        G_agg[u][v]['weight'] += w
                    else:
                        G_agg.add_edge(u, v, weight=w)
            # Add all nodes (0-48 for 7x7 grid)
            for n in range(49):
                if n not in G_agg:
                    G_agg.add_node(n)
            return G_agg

        G_win = aggregate_networks(win_networks)
        G_lose = aggregate_networks(lose_networks)

        # Grid layout for 7x7
        pos = {i: (i % 7, i // 7) for i in range(49)}

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, G, title in zip(axes, [G_win, G_lose], ['Winning Contexts', 'Losing Contexts']):
            ax.set_aspect('equal')
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=350, node_color='lightblue', edgecolors='black', linewidths=0.6)
            # Draw edges with thickness proportional to weight
            edges = G.edges(data=True)
            weights = [edge[2]['weight'] for edge in edges]
            max_weight = max(weights) if weights else 1
            for u, v, d in edges:
                nx.draw_networkx_edges(
                    G, pos, [(u, v)],
                    width=(d['weight'] / max_weight) * 3,
                    alpha=0.6,
                    ax=ax
                )
            # Draw zone labels
            for node_id, (x, y) in pos.items():
                ax.text(x, y, str(node_id), ha='center', va='center', fontsize=8, fontweight='bold', bbox=dict(boxstyle='circle', facecolor='white', alpha=0.9, linewidth=0.4))
            ax.set_title(title, fontweight='bold', fontsize=15)
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.axis('off')

        plt.suptitle('Passing Networks: Winning vs. Losing Contexts', fontsize=18, fontweight='bold')
        plt.tight_layout()
        if save_plot:
            out_path = self.output_dir / output_filename
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Side-by-side network plot saved to {out_path}")
        plt.show()
        return fig, axes
    """Comprehensive visualization for RQ1: Contextual Network Analysis"""
    
    def __init__(self, style='whitegrid', palette='Set2', figsize=(12, 8)):
        # Set style
        sns.set_style(style)
        sns.set_palette(palette)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        self.output_dir = Path("results/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define available metrics (check what actually exists in data)
        self.base_metrics = [
            'density', 'clustering_coefficient', 'avg_betweenness_centrality',
            'avg_eigenvector_centrality', 'avg_path_length', 'centralization'
        ]
    
    def create_all_rq1_plots(self, results_df, statistical_results, save_plots=True):
        """Create all RQ1 visualization plots"""
        print("Creating RQ1 visualizations...")
        print(f"Dataset shape: {results_df.shape}")
        print(f"Available columns: {list(results_df.columns)}")
        
        # Check what metrics are actually available
        available_metrics = [m for m in self.base_metrics if m in results_df.columns]
        print(f"Available metrics: {available_metrics}")
        
        # Check context distributions
        for context in ['score_context', 'phase_context', 'intensity_context']:
            if context in results_df.columns:
                print(f"{context} distribution:")
                print(results_df[context].value_counts())
                print()
        
        plots_created = []
        
        # 1. Data Overview
        plots_created.extend(self._plot_data_overview(results_df, save_plots))
        
        # 2. Context Comparisons - only for contexts with multiple categories
        plots_created.extend(self._plot_meaningful_comparisons(results_df, save_plots))
        
        # 3. Statistical Results Visualization
        plots_created.extend(self._plot_statistical_results(statistical_results, save_plots))
        
        # 4. Intensity Analysis (this shows the strongest effects)
        plots_created.extend(self._plot_intensity_analysis(results_df, save_plots))
        
        # 5. Temporal Patterns
        plots_created.extend(self._plot_temporal_analysis(results_df, save_plots))
        
        # 6. Correlation Analysis
        plots_created.extend(self._plot_correlation_analysis(results_df, save_plots))
        
        print(f"✅ Created {len(plots_created)} visualization plots")
        if save_plots:
            print(f"📁 Plots saved to: {self.output_dir}")
        
        return plots_created
    
    def _plot_data_overview(self, df, save_plots):
        """Create overview of the dataset"""
        plots = []
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sample sizes by context
        ax1 = axes[0, 0]
        contexts = ['score_context', 'phase_context', 'intensity_context']
        context_data = []
        
        for context in contexts:
            if context in df.columns:
                counts = df[context].value_counts()
                for label, count in counts.items():
                    context_data.append({
                        'Context Type': context.replace('_context', '').title(),
                        'Label': label,
                        'Count': count
                    })
        
        if context_data:
            context_df = pd.DataFrame(context_data)
            sns.barplot(data=context_df, x='Context Type', y='Count', hue='Label', ax=ax1)
            ax1.set_title('Sample Sizes by Context Type')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Metric distributions
        ax2 = axes[0, 1]
        key_metrics = ['density', 'clustering_coefficient']
        available_key_metrics = [m for m in key_metrics if m in df.columns]
        
        if available_key_metrics:
            for i, metric in enumerate(available_key_metrics):
                ax2.hist(df[metric], alpha=0.7, label=metric.replace('_', ' ').title(), bins=30)
            ax2.set_title('Distribution of Key Metrics')
            ax2.set_xlabel('Metric Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # 3. Pass count distribution
        ax3 = axes[1, 0]
        if 'pass_count' in df.columns:
            ax3.hist(df['pass_count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Distribution of Pass Counts per Window')
            ax3.set_xlabel('Number of Passes')
            ax3.set_ylabel('Frequency')
            ax3.axvline(df['pass_count'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["pass_count"].mean():.1f}')
            ax3.legend()
        
        # 4. Teams and matches overview
        ax4 = axes[1, 1]
        if 'team' in df.columns and 'match_id' in df.columns:
            team_counts = df['team'].value_counts().head(10)
            ax4.bar(range(len(team_counts)), team_counts.values, alpha=0.7)
            ax4.set_title(f'Top 10 Teams by Windows\n(Total: {df["team"].nunique()} teams, {df["match_id"].nunique()} matches)')
            ax4.set_xlabel('Team Rank')
            ax4.set_ylabel('Number of Windows')
        
        plt.tight_layout()
        
        if save_plots:
            filename = "01_data_overview.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plots.append(filename)
            plots.extend(self._save_subplots(fig, axes, "01_data_overview"))
        plt.show()
        
        return plots
    
    def _plot_meaningful_comparisons(self, df, save_plots):
        """Plot comparisons only for contexts with meaningful variation"""
        plots = []
        
        # Check which contexts have multiple categories with sufficient data
        meaningful_contexts = []
        
        for context in ['phase_context', 'intensity_context']:  # Skip score_context as it only has 'tied'
            if context in df.columns:
                counts = df[context].value_counts()
                if len(counts) > 1 and counts.min() >= 10:  # At least 2 categories with 10+ observations
                    meaningful_contexts.append(context)
        
        if not meaningful_contexts:
            print("No contexts with sufficient variation for comparison plots")
            return plots
        
        available_metrics = [m for m in self.base_metrics if m in df.columns]
        
        for context in meaningful_contexts:
            # Create comparison plots for this context
            n_metrics = len(available_metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, metric in enumerate(available_metrics):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                # Box plot with individual points
                sns.boxplot(data=df, x=context, y=metric, ax=ax)
                # Set transparency for all patches (boxes, whiskers, etc.)
                for patch in ax.patches:
                    patch.set_alpha(0.7)

                sns.stripplot(data=df, x=context, y=metric, ax=ax, 
                             size=3, alpha=0.3, color='black')
                
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel(context.replace('_', ' ').title())
                ax.tick_params(axis='x', rotation=45)
                
                # Add statistical annotation if significant
                if context.replace('_context', '_context') in ['phase_context', 'intensity_context']:
                    # Add mean values as text
                    means = df.groupby(context)[metric].mean()
                    for j, (label, mean_val) in enumerate(means.items()):
                        ax.text(j, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}', 
                               ha='center', va='top', fontweight='bold')
            
            # Hide empty subplots
            for i in range(len(available_metrics), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.suptitle(f'Network Metrics by {context.replace("_", " ").title()}', 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            
            if save_plots:
                filename = f"02_comparisons_{context}.png"
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
                plots.append(filename)
                plots.extend(self._save_subplots(fig, axes, f"02_comparisons_{context}"))
            plt.show()
        
        return plots
    
    def _plot_statistical_results(self, statistical_results, save_plots):
        """Visualize statistical test results"""
        plots = []
        
        # Create summary of statistical significance
        sig_data = []
        effect_data = []
        
        for context_type, results in statistical_results.items():
            if 'group_comparisons' in results:
                for metric, test_result in results['group_comparisons'].items():
                    if 'error' not in test_result:
                        sig_data.append({
                            'Context': context_type.replace('_', ' ').title(),
                            'Metric': metric.replace('_', ' ').title(),
                            'P_Value': test_result.get('p_value', 1.0),
                            'Significant': test_result.get('significant', False),
                            'Test': test_result.get('test_name', 'Unknown')
                        })
            
            if 'effect_sizes' in results:
                for metric, effect_info in results['effect_sizes'].items():
                    if 'eta_squared' in effect_info:
                        effect_data.append({
                            'Context': context_type.replace('_', ' ').title(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Eta_Squared': effect_info['eta_squared'],
                            'Interpretation': effect_info.get('interpretation', 'unknown')
                        })
        
        if sig_data and effect_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Significance heatmap
            sig_df = pd.DataFrame(sig_data)
            if not sig_df.empty:
                pivot_sig = sig_df.pivot(index='Metric', columns='Context', values='P_Value')
                sns.heatmap(pivot_sig, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                           ax=ax1, cbar_kws={'label': 'p-value'})
                ax1.set_title('Statistical Significance (p-values)')
                
                # Add significance markers
                for i in range(len(pivot_sig.index)):
                    for j in range(len(pivot_sig.columns)):
                        p_val = pivot_sig.iloc[i, j]
                        if not pd.isna(p_val) and p_val < 0.05:
                            ax1.text(j + 0.5, i + 0.7, '***', ha='center', va='center', 
                                   color='white', fontweight='bold', fontsize=12)
            
            # Effect sizes
            effect_df = pd.DataFrame(effect_data)
            if not effect_df.empty:
                pivot_effect = effect_df.pivot(index='Metric', columns='Context', values='Eta_Squared')
                sns.heatmap(pivot_effect, annot=True, fmt='.3f', cmap='viridis', 
                           ax=ax2, cbar_kws={'label': 'η² (Effect Size)'})
                ax2.set_title('Effect Sizes (η²)')
            
            plt.tight_layout()
            
            if save_plots:
                filename = "03_statistical_results.png"
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
                plots.append(filename)
                plots.extend(self._save_subplots(fig, (ax1, ax2), "03_statistical_results"))
            plt.show()
        
        return plots
    
    def _plot_intensity_analysis(self, df, save_plots):
        """Focus on intensity context which shows the strongest effects"""
        plots = []
        
        if 'intensity_context' not in df.columns:
            return plots
        
        # Check if we have variation in intensity
        intensity_counts = df['intensity_context'].value_counts()
        if len(intensity_counts) < 2:
            return plots
        
        available_metrics = [m for m in self.base_metrics if m in df.columns]
        
        # Create detailed intensity analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Violin plot for better distribution visualization
            sns.violinplot(data=df, x='intensity_context', y=metric, ax=ax)
            
            # Add box plot overlay
            sns.boxplot(data=df, x='intensity_context', y=metric, ax=ax, width=0.3)
            
            # Set transparency for violin and box plot elements
            for patch in ax.patches:
                patch.set_alpha(0.7)
            for collection in ax.collections:
                collection.set_alpha(0.7)
                
            ax.set_title(f'{metric.replace("_", " ").title()}\nby Match Intensity')
            ax.set_xlabel('Match Intensity')
            
            # Add sample sizes
            for j, intensity in enumerate(intensity_counts.index):
                count = intensity_counts[intensity]
                ax.text(j, ax.get_ylim()[0], f'n={count}', ha='center', va='bottom', 
                       fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Detailed Analysis: Network Metrics by Match Intensity\n(Strongest Effect Found)', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_plots:
            filename = "04_intensity_detailed.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plots.append(filename)
            plots.extend(self._save_subplots(fig, axes, "04_intensity_detailed"))
        plt.show()
        
        return plots
    
    def _plot_temporal_analysis(self, df, save_plots):
        """Analyze temporal patterns"""
        plots = []
        
        if 'start_minute' not in df.columns:
            return plots
        
        # Create time bins
        df_temp = df.copy()
        df_temp['time_bin'] = pd.cut(df_temp['start_minute'], 
                                    bins=[0, 15, 30, 45, 60, 75, 90], 
                                    labels=['0-15', '15-30', '30-45', '45-60', '60-75', '75-90'])
        
        available_metrics = [m for m in self.base_metrics[:4] if m in df.columns]  # Top 4 metrics
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                
                # Calculate means and confidence intervals
                temporal_stats = df_temp.groupby('time_bin')[metric].agg(['mean', 'std', 'count']).reset_index()
                temporal_stats['se'] = temporal_stats['std'] / np.sqrt(temporal_stats['count'])
                temporal_stats['ci_lower'] = temporal_stats['mean'] - 1.96 * temporal_stats['se']
                temporal_stats['ci_upper'] = temporal_stats['mean'] + 1.96 * temporal_stats['se']
                
                # Plot line with confidence interval
                x_pos = range(len(temporal_stats))
                ax.plot(x_pos, temporal_stats['mean'], marker='o', linewidth=2, markersize=8)
                ax.fill_between(x_pos, temporal_stats['ci_lower'], temporal_stats['ci_upper'], 
                               alpha=0.3, label='95% CI')
                
                ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
                ax.set_xlabel('Match Period (minutes)')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xticks(x_pos)
                ax.set_xticklabels(temporal_stats['time_bin'])
                ax.legend()
                
                # Add sample sizes
                for j, (_, row) in enumerate(temporal_stats.iterrows()):
                    ax.text(j, ax.get_ylim()[0], f'n={int(row["count"])}', 
                           ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Temporal Evolution of Network Metrics', fontsize=16, y=0.98)
            plt.tight_layout()
            
            if save_plots:
                filename = "05_temporal_patterns.png"
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
                plots.append(filename)
                plots.extend(self._save_subplots(fig, axes, "05_temporal_patterns"))
            plt.show()
        
        return plots
    
    def _plot_correlation_analysis(self, df, save_plots):
        """Analyze correlations between metrics"""
        plots = []
        
        available_metrics = [m for m in self.base_metrics if m in df.columns]
        
        if len(available_metrics) > 1:
            # Correlation matrix
            corr_matrix = df[available_metrics].corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                       cmap='RdBu_r', center=0, square=True,
                       xticklabels=[m.replace('_', ' ').title() for m in available_metrics],
                       yticklabels=[m.replace('_', ' ').title() for m in available_metrics])
            plt.title('Network Metrics Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            
            if save_plots:
                filename = "06_correlations.png"
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
                plots.append(filename)
                # Note: No individual subplots to save for correlation matrix
            plt.show()
        
        return plots

    def create_key_findings_summary(self, df, statistical_results, save_plots=True):
        """Create a summary plot highlighting key findings"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sample distribution
        if 'intensity_context' in df.columns:
            intensity_counts = df['intensity_context'].value_counts()
            colors = ['lightcoral', 'gold', 'lightgreen']
            wedges, texts, autotexts = ax1.pie(intensity_counts.values, 
                                              labels=intensity_counts.index,
                                              autopct='%1.1f%%', 
                                              colors=colors[:len(intensity_counts)])
            ax1.set_title('Distribution by Match Intensity\n(Key Finding: Intensity Matters!)')
        
        # 2. Key metric comparison
        if 'intensity_context' in df.columns and 'density' in df.columns:
            sns.barplot(data=df, x='intensity_context', y='density', ax=ax2, 
                       palette='viridis')
            # Set transparency for bars
            for patch in ax2.patches:
                patch.set_alpha(0.8)
            ax2.set_title('Network Density by Intensity\n(p < 0.001, η² = 0.388)')
            ax2.set_ylabel('Network Density')
            ax2.set_xlabel('Match Intensity')
        
        # 3. Effect sizes summary
        effect_data = []
        for context_type, results in statistical_results.items():
            if 'effect_sizes' in results:
                for metric, effect_info in results['effect_sizes'].items():
                    if 'eta_squared' in effect_info:
                        effect_data.append({
                            'Context': context_type.replace('_context', '').title(),
                            'Effect_Size': effect_info['eta_squared'],
                            'Interpretation': effect_info.get('interpretation', 'unknown')
                        })
        
        if effect_data:
            effect_df = pd.DataFrame(effect_data)
            sns.barplot(data=effect_df, x='Context', y='Effect_Size', ax=ax3, 
                       palette='Set2')
            # Set transparency for bars
            for patch in ax3.patches:
                patch.set_alpha(0.8)
            ax3.set_title('Effect Sizes by Context Type\n(η² values)')
            ax3.set_ylabel('Eta-squared (η²)')
            ax3.axhline(y=0.02, color='gray', linestyle='--', alpha=0.7, label='Small (0.02)')
            ax3.axhline(y=0.08, color='orange', linestyle='--', alpha=0.7, label='Medium (0.08)')
            ax3.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='Large (0.20)')
            ax3.legend()
        
        # 4. Significance summary
        sig_summary = []
        for context_type, results in statistical_results.items():
            if 'group_comparisons' in results:
                total = len(results['group_comparisons'])
                significant = sum(1 for test in results['group_comparisons'].values() 
                                if test.get('significant', False))
                sig_summary.append({
                    'Context': context_type.replace('_context', '').title(),
                    'Significant': significant,
                    'Total': total,
                    'Percentage': (significant/total)*100 if total > 0 else 0
                })
        
        if sig_summary:
            sig_df = pd.DataFrame(sig_summary)
            bars = ax4.bar(sig_df['Context'], sig_df['Percentage'], 
                          color=['lightcoral', 'gold', 'lightgreen'])
            # Set transparency for bars
            for bar in bars:
                bar.set_alpha(0.8)
            ax4.set_title('Percentage of Significant Tests\n(p < 0.05)')
            ax4.set_ylabel('% Significant Tests')
            ax4.set_ylim(0, 100)
            
            # Add percentage labels
            for bar, pct in zip(bars, sig_df['Percentage']):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                        f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('RQ1 Key Findings: Match Intensity Drives Network Structure Changes', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_plots:
            filename = "07_key_findings_summary.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.show()
            return filename
        
        plt.show()
        return None

    def _save_subplots(self, fig, axes, prefix):
        """
        Save each subplot from a figure individually by redrawing contents.
        
        Args:
            fig: matplotlib Figure object
            axes: single axis or array of axes
            prefix: prefix for filenames (e.g., '01_data_overview')
        """
        plots = []
        axes = np.atleast_1d(axes).flatten()

        for i, ax in enumerate(axes, start=1):
            if not ax.get_visible():
                continue

            # Create a new figure for this subplot
            fig_indiv, ax_indiv = plt.subplots(figsize=(6, 4))

            # --- Replot lines ---
            for line in ax.get_lines():
                ax_indiv.plot(*line.get_data(),
                            label=line.get_label(),
                            color=line.get_color(),
                            linestyle=line.get_linestyle(),
                            marker=line.get_marker())

            # --- Replot bars/hist patches (Rectangles) ---
            for patch in ax.patches:
                if hasattr(patch, "get_xy"):  # e.g., Rectangle
                    x, y = patch.get_xy()
                    width, height = patch.get_width(), patch.get_height()
                    ax_indiv.bar(x, height, width=width,
                                color=patch.get_facecolor(),
                                label=patch.get_label())

            # --- Titles, labels, legend ---
            ax_indiv.set_title(ax.get_title())
            ax_indiv.set_xlabel(ax.get_xlabel())
            ax_indiv.set_ylabel(ax.get_ylabel())

            # Handle legend - need to create new handles for the new figure
            handles, labels = ax.get_legend_handles_labels()
            if labels and any(label and label != '_nolegend_' for label in labels):
                # Filter out empty or default labels
                filtered_labels = []
                filtered_handles = []
                for handle, label in zip(handles, labels):
                    if label and label != '_nolegend_' and not label.startswith('_'):
                        filtered_labels.append(label)
                        filtered_handles.append(handle)
                
                if filtered_labels:
                    try:
                        ax_indiv.legend(filtered_labels, loc="best")
                    except (RuntimeError, ValueError):
                        # If legend creation fails, skip it
                        pass

            # Save individual file
            fname = f"{prefix}_{chr(96+i)}.png"  # → 01a, 01b, ...
            fig_indiv.savefig(self.output_dir / fname, dpi=300, bbox_inches='tight')
            plt.close(fig_indiv)
            plots.append(fname)

        return plots