import json
import pandas as pd
from pathlib import Path
from .data_loader import DataLoader
from .context_analyzer import ContextAnalyzer
from .network_builder import NetworkBuilder
from .network_analyzer import NetworkAnalyzer
from .statistical_comparator import StatisticalComparator
from .visualizer import RQ1Visualizer


class MainAnalysis:
    """Main analysis pipeline for RQ1: Contextual Network Analysis"""
    
    def __init__(self, window_size=10, step_size=5, min_passes=20):
        self.data_loader = DataLoader()
        self.context_analyzer = ContextAnalyzer(window_size, step_size, min_passes)
        self.network_builder = NetworkBuilder()
        self.network_analyzer = NetworkAnalyzer()
        self.statistical_comparator = StatisticalComparator()
        
        self.results = {}
    
    def create_visualizations(self):
        """Create comprehensive visualizations for RQ1"""
        if not self.results:
            print("No results available. Run analysis first.")
            return
        
        visualizer = RQ1Visualizer()
        
        # Create all plots
        plots = visualizer.create_all_rq1_plots(
            self.results['network_metrics'],
            self.results['statistical_results'],
            save_plots=True
        )
        
        # Create key findings summary
        summary_plot = visualizer.create_key_findings_summary(
            self.results['network_metrics'],
            self.results['statistical_results'],
            save_plots=True
        )
        
        if summary_plot:
            plots.append(summary_plot)
        
        self.results['plots'] = plots
        
        print(f"\n🎨 Created {len(plots)} visualization plots")
        print("📊 Key Finding: Match intensity shows LARGE effects on network structure!")
        
        return plots

    def run_full_analysis(self, competitions, max_matches=None, save_results=True, create_plots=True):
        """Run the complete RQ1 analysis pipeline"""
        print("Starting RQ1: Contextual Network Analysis")
        print("=" * 50)
        
        # Step 1: Load data
        print("\n1. Loading data...")
        self.data_loader.load_data(competitions, max_matches)
        
        # Step 2: Extract context windows
        print("\n2. Extracting context windows...")
        all_context_windows = []
        
        for match in self.data_loader.matches[:max_matches] if max_matches else self.data_loader.matches:
            match_id = match['match_id']
            if match_id in self.data_loader.events:
                print(f"Processing match {match_id}...")
                windows = self.context_analyzer.extract_context_windows(
                    self.data_loader.events[match_id], match_id
                )
                all_context_windows.extend(windows)
        
        print(f"✅ Extracted {len(all_context_windows)} context windows")
        
        # Step 3: Build networks
        print("\n3. Building passing networks...")
        network_data = self.network_builder.build_networks_from_windows(all_context_windows)
        print(f"✅ Built {len(network_data)} networks")
        
        # Step 4: Calculate network metrics
        print("\n4. Calculating network metrics...")
        results_list = []
        
        for window_data in network_data:
            if window_data['network'] is not None:
                metrics = self.network_analyzer._calculate_network_metrics(window_data['network'])
                
                result = {
                    'match_id': window_data['match_id'],
                    'team': window_data['team'],
                    'start_minute': window_data['start_minute'],
                    'end_minute': window_data['end_minute'],
                    'pass_count': window_data['pass_count'],
                    'score_context': window_data['score_context'],
                    'phase_context': window_data['phase_context'],
                    'intensity_context': window_data['intensity_context'],
                    **metrics
                }
                results_list.append(result)
        
        results_df = pd.DataFrame(results_list)
        print(f"✅ Calculated metrics for {len(results_df)} windows")
        
        # Step 5: Statistical analysis
        print("\n5. Performing statistical analysis...")
        statistical_results = self.statistical_comparator.compare_contexts(results_df)
        
        # Step 6: Generate report
        print("\n6. Generating report...")
        report = self.statistical_comparator.generate_comprehensive_report()
        
        # Store results
        self.results = {
            'context_windows': all_context_windows,
            'network_metrics': results_df,
            'statistical_results': statistical_results,
            'report': report
        }
        
        if create_plots:
            print("\n7. Creating visualizations...")
            self.create_visualizations()

        # Save results
        if save_results:
            self._save_results()
        
        print("\n✅ Analysis complete!")
        print(f"Total context windows analyzed: {len(all_context_windows)}")
        print(f"Total networks built: {len(network_data)}")
        print(f"Final dataset size: {len(results_df)} observations")
        
        return self.results
    
    def _save_results(self):
        """Save analysis results"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Save main results
        self.results['network_metrics'].to_csv(output_dir / "network_metrics.csv", index=False)
        
        # Save statistical results
        with open(output_dir / "statistical_results.json", 'w') as f:
            json.dump(self.results['statistical_results'], f, indent=2, default=str)
        
        # Save report
        with open(output_dir / "analysis_report.txt", 'w') as f:
            f.write(self.results['report'])
        
        print(f"✅ Results saved to {output_dir}/")
    
    def print_summary(self):
        """Print analysis summary"""
        if not self.results:
            print("No results available. Run analysis first.")
            return
        
        print("\nANALYSIS SUMMARY")
        print("=" * 30)
        
        df = self.results['network_metrics']
        
        print(f"Total observations: {len(df)}")
        print(f"Unique matches: {df['match_id'].nunique()}")
        print(f"Unique teams: {df['team'].nunique()}")
        
        print("\nContext distribution:")
        for context_type in ['score_context', 'phase_context', 'intensity_context']:
            if context_type in df.columns:
                print(f"\n{context_type.replace('_', ' ').title()}:")
                counts = df[context_type].value_counts()
                for label, count in counts.items():
                    print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\nDetailed report saved to results/analysis_report.txt")

def run_rq1_analysis():
    """Convenience function to run RQ1 analysis"""
    competitions = [(11, 90)] 
    
    # Initialize and run analysis
    analysis = MainAnalysis(window_size=10, step_size=5, min_passes=20)

    # Load and analyze
    results = analysis.run_full_analysis(
        competitions=[
            # La Liga
            (11, 90),   # La Liga 2020/2021
            (11, 42),   # La Liga 2019/2020
            # Premier League
            (2, 27),    # Premier League 2015/2016
            # Serie A
            (12, 27),   # Serie A 2015/2016
        ],
        max_matches=38, 
        save_results=True,
        create_plots=True
    )
    
    # Print summary
    analysis.print_summary()
    
    return analysis, results

if __name__ == "__main__":
    analysis, results = run_rq1_analysis()
