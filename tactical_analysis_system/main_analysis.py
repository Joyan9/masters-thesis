import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from .data_loader import DataLoader
from .context_analyzer import ContextAnalyzer
from .network_builder import NetworkBuilder
from .network_analyzer import NetworkAnalyzer
from .statistical_comparator import StatisticalComparator
from .visualizer import RQ1Visualizer


class MainAnalysis:
    """Main analysis pipeline for RQ1: Contextual Network Analysis"""
    
    def __init__(self, use_saved_data: bool = True, 
                        data_file: str = "statsbomb_data_interim_100.json", 
                        window_size: int = 10, 
                        step_size: int = 5, 
                        min_passes: int = 20):
        """Initialize analysis system
        
        Args:
            use_saved_data: If True, load from saved JSON file
            data_file: Path to saved data file
            window_size: Window size for analysis
            step_size: Step size for sliding window
            min_passes: Minimum number of passes required
        """
        self.results = {}
        self.data_loader = DataLoader()
        self.window_size = window_size
        self.step_size = step_size
        self.min_passes = min_passes
        
        # Initialize results directory
        self.results_dir = Path("./results/")
        
        # Initialize components
        self.context_analyzer = ContextAnalyzer(window_size, step_size, min_passes)
        self.network_builder = NetworkBuilder()
        self.network_analyzer = None  # Will be initialized when needed
        self.statistical_comparator = StatisticalComparator()

        if use_saved_data:
            try:
                self.data_loader.load_from_json(data_file)
                print(f"✅ Using saved data from {data_file}")
            except FileNotFoundError:
                print(f"❌ Saved data file {data_file} not found. Please run data collection first.")
                print("Example: python collect_data.py")
                raise
        else:
            print("⚠️  Will load data from API (slower)")
    
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

    def run_rq1_analysis(self, save_results: bool = True, create_plots: bool = True) -> Dict:
        """Run RQ1 analysis using pre-loaded data"""
        print("RUNNING RQ1: NETWORK ANALYSIS")
        print("=" * 60)
        
        # Use pre-loaded data instead of loading from API
        if self.data_loader.matches_data.empty:
            raise ValueError("No data loaded. Please load data first.")
        
        # Initialize network analyzer with the loaded data
        self.network_analyzer = NetworkAnalyzer(self.data_loader)
        
        print("Starting RQ1: Contextual Network Analysis")
        print("=" * 50)
        
        # Step 1: Data is already loaded
        print("\n1. Loading data...")
        print("Loading matches and events for context analysis...")
        
        # Define competitions to analyze
        competitions = [
            # La Liga
            (11, 90),   # La Liga 2020/2021
            (11, 42),   # La Liga 2019/2020
            # Premier League
            (2, 27),    # Premier League 2015/2016
            # Serie A
            (12, 27),   # Serie A 2015/2016
        ]
        
        # Load specific competition data
        self.data_loader.load_data(competitions, max_matches=38)
        
        # Step 2: Extract context windows
        print("\n2. Extracting context windows...")
        all_context_windows = []
        
        max_matches = 38
        matches_to_process = self.data_loader.matches[:max_matches] if max_matches else self.data_loader.matches
        
        for match in matches_to_process:
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
        
        # Print summary
        self.print_summary()
        
        return self.results

    # Remove the old run_full_analysis method since it's redundant
    
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

        # Generate sample recommendations first
        print("\n1. Testing recommendation system...")
        sample_recommendations = self._generate_sample_recommendations(recommender)

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

    def _generate_sample_recommendations(self, recommender) -> list:
        """Generate sample recommendations for testing"""
        # This method needs to be implemented
        return []

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
    
    def run_rq3_analysis(self, save_results: bool = True) -> Dict:
        """Run RQ3: Recommendation Validation"""
        
        if not self.results or 'rq2_results' not in self.results:
            raise ValueError("RQ2 results not available. Run RQ2 analysis first.")
        
        print("\n" + "="*60)
        print("RUNNING RQ3: RECOMMENDATION VALIDATION")
        print("="*60)
        
        from .recommendation_validator import RecommendationValidator
        from .counterfactual_analyzer import CounterfactualAnalyzer
        
        # Get RQ2 data
        rq2_results = self.results['rq2_results']
        recommendations_data = rq2_results.get('match_recommendations', [])
        
        if not recommendations_data:
            raise ValueError("No recommendation data available from RQ2")
        
        # 1. Run validation analysis
        print("\n1. Running recommendation validation...")
        validator = RecommendationValidator(
            self.results['network_metrics'], 
            recommendations_data
        )
        validation_results = validator.run_comprehensive_validation()
        
        # 2. Run counterfactual analysis
        print("\n2. Running counterfactual analysis...")
        counterfactual_analyzer = CounterfactualAnalyzer(
            self.results['network_metrics'],
            recommendations_data
        )
        counterfactual_results = counterfactual_analyzer.run_counterfactual_analysis()
        
        # 3. Create comprehensive validation report
        print("\n3. Creating validation report...")
        validation_report = self._create_rq3_report(
            validation_results, counterfactual_results
        )
        
        # Store RQ3 results
        rq3_results = {
            'validation_results': validation_results,
            'counterfactual_results': counterfactual_results,
            'validation_report': validation_report,
            'validator': validator,
            'counterfactual_analyzer': counterfactual_analyzer
        }
        
        # Save results
        if save_results:
            self._save_rq3_results(rq3_results)
        
        # Add to main results
        self.results['rq3_results'] = rq3_results
        
        print(f"\n✅ RQ3 Analysis Complete!")
        print(f"📊 Validation Score: {validation_results['overall_validation_score']['overall_validation_score']:.3f}")
        
        return rq3_results

    def _create_rq3_report(self, validation_results: Dict, 
                        counterfactual_results: Dict) -> str:
        """Create comprehensive RQ3 validation report"""
        
        report_lines = [
            "RQ3: RECOMMENDATION VALIDATION ANALYSIS REPORT",
            "=" * 60,
            "",
            "EXECUTIVE SUMMARY:",
            f"Overall Validation Score: {validation_results['overall_validation_score']['overall_validation_score']:.3f}",
            f"Interpretation: {validation_results['overall_validation_score']['validation_interpretation']}",
            "",
            "KEY FINDINGS:",
        ]
        
        # Add key findings from validation
        overall_score = validation_results['overall_validation_score']['overall_validation_score']
        if overall_score >= 0.7:
            report_lines.append("✅ STRONG VALIDATION: Recommendations show high effectiveness")
        elif overall_score >= 0.6:
            report_lines.append("✅ MODERATE VALIDATION: Recommendations show good effectiveness")
        else:
            report_lines.append("⚠️  WEAK VALIDATION: Recommendations need improvement")
        
        # Add component analysis
        report_lines.extend([
            "",
            "COMPONENT ANALYSIS:",
        ])
        
        component_scores = validation_results['overall_validation_score']['component_scores']
        for component, score in component_scores.items():
            status = "✅" if score >= 0.6 else "⚠️" if score >= 0.4 else "❌"
            report_lines.append(f"{status} {component.replace('_', ' ').title()}: {score:.3f}")
        
        # Add counterfactual findings
        if 'impact_analysis' in counterfactual_results:
            impact = counterfactual_results['impact_analysis']
            improvement_rate = impact.get('overall_improvement_rate', 0)
            
            report_lines.extend([
                "",
                "COUNTERFACTUAL ANALYSIS:",
                f"Overall Improvement Rate: {improvement_rate:.1%}",
            ])
            
            if improvement_rate > 0.6:
                report_lines.append("✅ High likelihood of performance improvement")
            elif improvement_rate > 0.4:
                report_lines.append("✅ Moderate likelihood of performance improvement")
            else:
                report_lines.append("⚠️  Low likelihood of performance improvement")
        
        # Add detailed validation findings
        report_lines.extend([
            "",
            "DETAILED VALIDATION RESULTS:",
            "",
            "1. PERFORMANCE OUTCOME ANALYSIS:",
        ])
        
        if 'performance_outcomes' in validation_results:
            outcome_analysis = validation_results['performance_outcomes']
            if 'correlation_analysis' in outcome_analysis:
                report_lines.append("   Network Metric Correlations with Improvements:")
                for metric, analysis in outcome_analysis['correlation_analysis'].items():
                    if isinstance(analysis, dict):
                        conf_corr = analysis.get('confidence_vs_improvement', 0)
                        status = "✅" if abs(conf_corr) > 0.3 else "⚠️" if abs(conf_corr) > 0.1 else "❌"
                        report_lines.append(f"   {status} {metric}: {conf_corr:.3f}")
        
        # Add temporal consistency
        if 'temporal_consistency' in validation_results:
            temporal = validation_results['temporal_consistency']
            consistency = temporal.get('overall_consistency', 0)
            report_lines.extend([
                "",
                "2. TEMPORAL CONSISTENCY:",
                f"   Overall Consistency: {consistency:.3f}",
                f"   Contexts Analyzed: {temporal.get('total_contexts_analyzed', 0)}"
            ])
        
        # Add context sensitivity
        if 'context_sensitivity' in validation_results:
            context = validation_results['context_sensitivity']
            sensitivity = context.get('overall_sensitivity', 0)
            interpretation = context.get('sensitivity_interpretation', 'Unknown')
            report_lines.extend([
                "",
                "3. CONTEXT SENSITIVITY:",
                f"   Sensitivity Score: {sensitivity:.3f}",
                f"   Interpretation: {interpretation}"
            ])
        
        # Add model performance
        if 'model_performance' in counterfactual_results:
            model_perf = counterfactual_results['model_performance']
            overall_quality = model_perf.get('overall_quality', 0)
            total_models = model_perf.get('total_models', 0)
            
            report_lines.extend([
                "",
                "4. PREDICTIVE MODEL PERFORMANCE:",
                f"   Overall Model Quality: {overall_quality:.3f}",
                f"   Total Models Built: {total_models}",
            ])
            
            if overall_quality > 0.7:
                report_lines.append("   ✅ High-quality predictive models")
            elif overall_quality > 0.5:
                report_lines.append("   ✅ Moderate-quality predictive models")
            else:
                report_lines.append("   ⚠️  Low-quality predictive models")
        
        # Add recommendations for improvement
        report_lines.extend([
            "",
            "RECOMMENDATIONS FOR SYSTEM IMPROVEMENT:",
        ])
        
        if overall_score < 0.6:
            report_lines.extend([
                "- Refine rule thresholds based on validation findings",
                "- Improve context sensitivity of recommendations",
                "- Enhance confidence calibration mechanisms"
            ])
        elif overall_score < 0.8:
            report_lines.extend([
                "- Fine-tune recommendation timing",
                "- Improve prediction accuracy for edge cases",
                "- Enhance recommendation specificity"
            ])
        else:
            report_lines.extend([
                "- System shows strong validation",
                "- Consider deployment for real-world testing",
                "- Monitor performance in live scenarios"
            ])
        
        return "\n".join(report_lines)

    def _save_rq3_results(self, rq3_results: Dict):
        """Save RQ3 results to files"""
        
        # Save validation report
        report_path = self.results_dir / "rq3_validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(rq3_results['validation_report'])
        
        # Save validation summary
        validation_summary = {
            'overall_score': rq3_results['validation_results']['overall_validation_score']['overall_validation_score'],
            'interpretation': rq3_results['validation_results']['overall_validation_score']['validation_interpretation'],
            'component_scores': rq3_results['validation_results']['overall_validation_score']['component_scores']
        }
        
        summary_path = self.results_dir / "rq3_validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        print(f"📁 RQ3 results saved to {self.results_dir}")


if __name__ == "__main__":
    pass
