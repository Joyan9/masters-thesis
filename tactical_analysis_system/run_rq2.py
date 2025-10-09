from tactical_analysis_system.main_analysis import MainAnalysis
from tactical_analysis_system.data_loader import DataLoader
from tactical_analysis_system.visualizer import RQ1Visualizer
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 1. Rule Triggering Analysis - Show what's actually happening
def plot_rule_triggering_analysis(flat_recs, output_dir="results/plots/rq2"):
    ensure_dir(output_dir)
    
    # Analyze recommendation vs no-recommendation windows
    has_rec = [len(rec['recommendations']) > 0 for rec in flat_recs]
    trigger_counts = pd.Series(has_rec).value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall triggering rate
    labels = ['No Recommendations', 'Has Recommendations']
    colors = ['#e74c3c', '#2ecc71']
    axes[0].pie(trigger_counts.values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[0].set_title('Overall Rule Triggering Rate')
    
    # Primary focus distribution
    focus_counts = pd.Series([rec['summary']['primary_focus'] for rec in flat_recs]).value_counts()
    axes[1].barh(focus_counts.index, focus_counts.values, color=['#95a5a6', '#3498db'])
    axes[1].set_xlabel('Count')
    axes[1].set_title('Primary Focus Distribution')
    axes[1].set_ylabel('Focus Type')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_rule_triggering_analysis.png", dpi=300)
    plt.close()

# 2. Context Analysis - When are recommendations (not) made?
def plot_context_triggering_patterns(flat_recs, output_dir="results/plots/rq2"):
    ensure_dir(output_dir)
    
    df = pd.DataFrame([{
        'has_recommendations': len(rec['recommendations']) > 0,
        'score_context': rec['current_context']['score_context'],
        'phase_context': rec['current_context']['phase_context'],
        'intensity_context': rec['current_context']['intensity_context'],
        'urgency': rec['summary']['urgency'],
        'primary_focus': rec['summary']['primary_focus']
    } for rec in flat_recs])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # By Score Context
    score_cross = pd.crosstab(df['score_context'], df['has_recommendations'], normalize='index') * 100
    score_cross.plot(kind='bar', ax=axes[0,0], color=['#e74c3c', '#2ecc71'])
    axes[0,0].set_title('Recommendation Rate by Score Context')
    axes[0,0].set_ylabel('Percentage (%)')
    axes[0,0].legend(['No Rec', 'Has Rec'])
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
    
    # By Phase Context
    phase_cross = pd.crosstab(df['phase_context'], df['has_recommendations'], normalize='index') * 100
    phase_cross.plot(kind='bar', ax=axes[0,1], color=['#e74c3c', '#2ecc71'])
    axes[0,1].set_title('Recommendation Rate by Phase Context')
    axes[0,1].set_ylabel('Percentage (%)')
    axes[0,1].legend(['No Rec', 'Has Rec'])
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # By Intensity Context
    intensity_cross = pd.crosstab(df['intensity_context'], df['has_recommendations'], normalize='index') * 100
    intensity_cross.plot(kind='bar', ax=axes[1,0], color=['#e74c3c', '#2ecc71'])
    axes[1,0].set_title('Recommendation Rate by Intensity Context')
    axes[1,0].set_ylabel('Percentage (%)')
    axes[1,0].legend(['No Rec', 'Has Rec'])
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
    
    # By Urgency Level
    urgency_cross = pd.crosstab(df['urgency'], df['has_recommendations'], normalize='index') * 100
    urgency_cross.plot(kind='bar', ax=axes[1,1], color=['#e74c3c', '#2ecc71'])
    axes[1,1].set_title('Recommendation Rate by Urgency Level')
    axes[1,1].set_ylabel('Percentage (%)')
    axes[1,1].legend(['No Rec', 'Has Rec'])
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_context_triggering_patterns.png", dpi=300)
    plt.close()

# 3. Network Metrics When Rules Trigger vs Don't Trigger
def plot_metrics_at_triggering(flat_recs, output_dir="results/plots/rq2"):
    ensure_dir(output_dir)
    
    df = pd.DataFrame([{
        'has_recommendations': 'Yes' if len(rec['recommendations']) > 0 else 'No',
        **rec['current_metrics']
    } for rec in flat_recs])
    
    metrics = ['density', 'clustering_coefficient', 'avg_betweenness_centrality', 
               'centralization', 'avg_path_length']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        sns.boxplot(data=df, x='has_recommendations', y=metric, ax=axes[idx], 
                   palette={'No': '#e74c3c', 'Yes': '#2ecc71'})
        axes[idx].set_title(f'{metric.replace("_", " ").title()}')
        axes[idx].set_xlabel('Has Recommendations')
        axes[idx].set_ylabel('Value')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_metrics_at_triggering.png", dpi=300)
    plt.close()

# 4. Urgency Factors Analysis
def plot_urgency_factors(flat_recs, output_dir="results/plots/rq2"):
    ensure_dir(output_dir)
    
    # Extract all urgency factors
    all_factors = []
    for rec in flat_recs:
        factors = rec['situation_analysis'].get('urgency_factors', [])
        all_factors.extend(factors)
    
    if all_factors:
        factor_counts = pd.Series(all_factors).value_counts()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=factor_counts.values, y=factor_counts.index, palette='Reds_r')
        plt.xlabel('Frequency')
        plt.ylabel('Urgency Factor')
        plt.title('Figure 4: Most Common Urgency Factors Identified')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure4_urgency_factors.png", dpi=300)
        plt.close()
    else:
        print("No urgency factors found in recommendations.")

# 5. Temporal Consistency Analysis
def plot_temporal_consistency(flat_recs, output_dir="results/plots/rq2"):
    ensure_dir(output_dir)
    
    df = pd.DataFrame([{
        'match_id': rec['window_info']['match_id'],
        'minute': rec['window_info']['start_minute'],
        'temporal_consistency': rec['temporal_consistency'],
        'has_recommendations': len(rec['recommendations']) > 0
    } for rec in flat_recs])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temporal consistency distribution
    axes[0].hist(df['temporal_consistency'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Temporal Consistency Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Temporal Consistency')
    axes[0].axvline(df['temporal_consistency'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["temporal_consistency"].mean():.2f}')
    axes[0].legend()
    
    # Consistency by recommendation status
    # Convert boolean to string for palette mapping
    df['rec_status'] = df['has_recommendations'].map({False: 'No', True: 'Yes'})
    sns.boxplot(data=df, x='rec_status', y='temporal_consistency', ax=axes[1],
               palette={'No': '#e74c3c', 'Yes': '#2ecc71'})
    axes[1].set_xlabel('Has Recommendations')
    axes[1].set_ylabel('Temporal Consistency')
    axes[1].set_title('Temporal Consistency by Recommendation Status')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure5_temporal_consistency.png", dpi=300)
    plt.close()

# 6. Match-Level Summary
def plot_match_level_summary(flat_recs, output_dir="results/plots/rq2"):
    ensure_dir(output_dir)
    
    df = pd.DataFrame([{
        'match_id': rec['window_info']['match_id'],
        'team': rec['window_info']['team'],
        'has_recommendations': len(rec['recommendations']) > 0
    } for rec in flat_recs])
    
    match_summary = df.groupby(['match_id', 'team']).agg({
        'has_recommendations': ['sum', 'count']
    }).reset_index()
    match_summary.columns = ['match_id', 'team', 'rec_count', 'total_windows']
    match_summary['rec_rate'] = (match_summary['rec_count'] / match_summary['total_windows']) * 100
    
    plt.figure(figsize=(12, 6))
    x_labels = [f"{row['team'][:15]}\n({row['match_id']})" 
                for _, row in match_summary.iterrows()]
    colors = ['#2ecc71' if rate > 0 else '#e74c3c' for rate in match_summary['rec_rate']]
    
    plt.bar(range(len(match_summary)), match_summary['rec_rate'], color=colors, alpha=0.7, edgecolor='black')
    plt.xticks(range(len(match_summary)), x_labels, rotation=45, ha='right')
    plt.xlabel('Match (Team)')
    plt.ylabel('Recommendation Rate (%)')
    plt.title('Figure 6: Recommendation Rate by Match')
    plt.axhline(match_summary['rec_rate'].mean(), color='blue', linestyle='--', 
                label=f'Average: {match_summary["rec_rate"].mean():.1f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure6_match_summary.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    window_size = 10
    data_file = "statsbomb_data_interim_100.json"
    analysis = MainAnalysis(use_saved_data=True, data_file=data_file, window_size=window_size)

    print("1. Running RQ1 Analysis")
    rq1_results = analysis.run_rq1_analysis(max_matches=100, save_results=True, filepath=data_file)
    print("RQ1 Done")

    print("\n2. Running RQ2 Analysis")
    rq2_results = analysis.run_rq2_analysis(save_results=True)
    print("RQ2 Done")

    # --- RQ2 Plotting ---
    if 'match_recommendations' in rq2_results:
        recommendations = rq2_results['match_recommendations']
    else:
        raise ValueError("No match recommendations found in rq2_results")

    # Flatten all window-level recommendations into a single list
    flat_recs = []
    for match in recommendations:
        if 'window_recommendations' in match:
            flat_recs.extend(match['window_recommendations'])

    if not flat_recs:
        raise ValueError("No window-level recommendations found.")

    print(f"\n=== RQ2 PLOTTING SUMMARY ===")
    print(f"Total windows analyzed: {len(flat_recs)}")
    print(f"Windows with recommendations: {sum(len(r['recommendations']) > 0 for r in flat_recs)}")
    print(f"Windows without recommendations: {sum(len(r['recommendations']) == 0 for r in flat_recs)}")
    
    output_dir = "results/plots/rq2"
    
    print("\nGenerating improved visualizations...")
    print("1. Rule Triggering Analysis...")
    plot_rule_triggering_analysis(flat_recs, output_dir)
    
    print("2. Context Triggering Patterns...")
    plot_context_triggering_patterns(flat_recs, output_dir)
    
    print("3. Metrics at Triggering...")
    plot_metrics_at_triggering(flat_recs, output_dir)
    
    print("4. Urgency Factors...")
    plot_urgency_factors(flat_recs, output_dir)
    
    print("5. Temporal Consistency...")
    plot_temporal_consistency(flat_recs, output_dir)
    
    #print("6. Match-Level Summary...")
    #plot_match_level_summary(flat_recs, output_dir)
    
    print(f"\n✓ All plots saved to {output_dir}/")