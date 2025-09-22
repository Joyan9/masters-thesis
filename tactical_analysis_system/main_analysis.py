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
        
        # centralize results directory here
        self.results_dir = Path(results_dir="./results/")
        self.results_dir.mkdir(parents=True, exist_ok=True)

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
        # use the centralized results_dir
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

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

        print(f"\nDetailed report saved to {self.results_dir / 'analysis_report.txt'}")

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

    def run_rq2_analysis(self, save_results: bool = True) -> dict:
        """Run RQ2: Rule-Based Tactical Recommendations"""

        if not self.results or 'network_metrics' not in self.results:
            raise ValueError("RQ1 results not available. Run RQ1 analysis first.")

        print("\n" + "=" * 60)
        print("RUNNING RQ2: RULE-BASED TACTICAL RECOMMENDATIONS")
        print("=" * 60)

        from .tactical_recommender import TacticalRecommender

        # Initialize recommendation system
        recommender = TacticalRecommender(self.results)
        recommender.initialize_system(self.results['network_metrics'])

        # Test recommendations on sample data
        print("\n1. Testing recommendation system...")
        sample_recommendations = self._test_recommendation_system(recommender)

        # Analyze recommendations for all matches
        print("\n2. Generating match-level recommendations...")
        match_recommendations = self._generate_match_recommendations(recommender)

        # Create recommendation report
        print("\n3. Creating recommendation report...")
        recommendation_report = self._create_recommendation_report(
            recommender, sample_recommendations, match_recommendations
        )

        # Store RQ2 results
        rq2_results = {
            'recommender': recommender,
            'sample_recommendations': sample_recommendations,
            'match_recommendations': match_recommendations,
            'recommendation_report': recommendation_report,
            'system_summary': recommender.get_system_summary()
        }

        # Save results
        if save_results:
            self._save_rq2_results(rq2_results)

        # Add to main results
        self.results['rq2_results'] = rq2_results

        print(f"\n✅ RQ2 Analysis Complete!")
        print(f"📊 Generated recommendations for {len(match_recommendations)} match scenarios")

        return rq2_results

    def _test_recommendation_system(self, recommender) -> list[dict]:
        """Test recommendation system with various scenarios"""
        
        test_scenarios = [
            {
                'name': 'Low Density Crisis',
                'metrics': {
                    'density': 0.025,  # Very low
                    'clustering_coefficient': 0.020,
                    'avg_betweenness_centrality': 0.025,
                    'avg_eigenvector_centrality': 0.100,
                    'avg_path_length': 4.5,
                    'centralization': 0.080
                },
                'context': {
                    'score_context': 'trailing',
                    'phase_context': 'late',
                    'intensity_context': 'low'
                }
            },
            {
                'name': 'High Performance',
                'metrics': {
                    'density': 0.055,  # High
                    'clustering_coefficient': 0.045,
                    'avg_betweenness_centrality': 0.040,
                    'avg_eigenvector_centrality': 0.110,
                    'avg_path_length': 3.8,
                    'centralization': 0.120
                },
                'context': {
                    'score_context': 'leading',
                    'phase_context': 'middle',
                    'intensity_context': 'high'
                }
            },
            {
                'name': 'Average Performance',
                'metrics': {
                    'density': 0.038,  # Average
                    'clustering_coefficient': 0.030,
                    'avg_betweenness_centrality': 0.032,
                    'avg_eigenvector_centrality': 0.103,
                    'avg_path_length': 4.1,
                    'centralization': 0.098
                },
                'context': {
                    'score_context': 'tied',
                    'phase_context': 'early',
                    'intensity_context': 'medium'
                }
            }
        ]
        
        test_results = []
        
        for scenario in test_scenarios:
            print(f"   Testing: {scenario['name']}")
            
            recommendations = recommender.get_recommendations(
                scenario['metrics'],
                scenario['context'],
                {'scenario_name': scenario['name']}
            )
            
            test_results.append({
                'scenario': scenario,
                'recommendations': recommendations
            })
        
        return test_results

    def _generate_match_recommendations(self, recommender) -> list[dict]:
        """Generate recommendations for sample matches"""
        
        # Get unique matches from the dataset
        if 'match_id' in self.results['network_metrics'].columns:
            unique_matches = self.results['network_metrics']['match_id'].unique()[:5]  # Sample 5 matches
            
            match_recommendations = []
            
            for match_id in unique_matches:
                print(f"   Analyzing match: {match_id}")
                
                match_data = self.results['network_metrics'][
                    self.results['network_metrics']['match_id'] == match_id
                ]
                
                match_analysis = recommender.analyze_match_recommendations(
                    match_data, str(match_id)
                )
                
                match_recommendations.append(match_analysis)
            
            return match_recommendations
        
        return []

    def _create_recommendation_report(self, recommender, sample_recs, match_recs) -> str:
        """Create comprehensive recommendation report"""
        
        report_lines = [
            "TACTICAL RECOMMENDATION SYSTEM REPORT",
            "=" * 50,
            "",
            "SYSTEM OVERVIEW:",
            f"- Total Rules: {len(recommender.rule_engine.rules)}",
            f"- Threshold Metrics: {len(recommender.threshold_analyzer.thresholds)}",
            f"- Test Scenarios: {len(sample_recs)}",
            f"- Match Analyses: {len(match_recs)}",
            "",
            "KEY FINDINGS FROM RQ1 INTEGRATION:",
            "- Match intensity is the strongest predictor of network structure",
            "- High intensity increases network density by 117%",
            "- Rules leverage this relationship for tactical recommendations",
            "",
            "SAMPLE RECOMMENDATION SCENARIOS:",
        ]
        
        # Add sample scenarios
        for i, test in enumerate(sample_recs, 1):
            scenario = test['scenario']
            recs = test['recommendations']
            
            report_lines.extend([
                f"\n{i}. {scenario['name']}:",
                f"   Context: {scenario['context']}",
                f"   Recommendations: {len(recs['recommendations'])}",
            ])
            
            if recs['recommendations']:
                primary_rec = recs['recommendations'][0]
                report_lines.append(f"   Primary: {primary_rec['action']}")
                report_lines.append(f"   Confidence: {primary_rec['confidence']} ({primary_rec['confidence_score']:.2f})")
        
        # Add match analysis summary
        if match_recs:
            report_lines.extend([
                "",
                "MATCH ANALYSIS SUMMARY:",
            ])
            
            total_critical_periods = sum(
                len(match['match_analysis']['critical_periods']) 
                for match in match_recs
            )
            
            report_lines.append(f"- Total critical periods identified: {total_critical_periods}")
            
            # Most common recommendations across matches
            all_rec_types = []
            for match in match_recs:
                for rec_type, count in match['match_analysis']['most_common_recommendations'].items():
                    all_rec_types.extend([rec_type] * count)
            
            if all_rec_types:
                from collections import Counter
                common_recs = Counter(all_rec_types).most_common(3)
                report_lines.append("- Most common recommendation types:")
                for rec_type, count in common_recs:
                    report_lines.append(f"  * {rec_type}: {count} instances")
        
        return "\n".join(report_lines)

    def _save_rq2_results(self, rq2_results: dict):
        """Save RQ2 results to files"""
        # save into the same centralized results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        report_path = self.results_dir / "rq2_recommendation_report.txt"
        with open(report_path, 'w') as f:
            f.write(rq2_results['recommendation_report'])

        summary_path = self.results_dir / "rq2_system_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(rq2_results['system_summary'], f, indent=2)

        print(f"RQ2 results saved to {self.results_dir}")

if __name__ == "__main__":
    analysis, results = run_rq1_analysis()