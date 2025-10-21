import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from .data_loader import DataLoader
from .context_analyzer import ContextAnalyzer
from .network_builder import NetworkBuilder
from .network_analyzer import NetworkAnalyzer
from .statistical_comparator import StatisticalComparator
from .tactical_recommender import TacticalRecommender
from .recommendation_validator import RecommendationValidator
from .counterfactual_analyzer import CounterfactualAnalyzer

class MainAnalysis:
    """Main analysis system for tactical analysis in football"""
    
    def __init__(self, use_saved_data: bool = True, 
                        data_file: str = "statsbomb_data_interim_100.json", 
                        window_size: int = 5, 
                        step_size: int = 1, 
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

    def run_rq1_analysis(self, 
                           max_matches: int = 100,
                           save_results: bool = True,  
                           filepath: str = "statsbomb_data_interim_100.json") -> Dict:
            
            """Run RQ1: Contextual Network Analysis"""
            
            print("RUNNING RQ1: NETWORK ANALYSIS")
            print("=" * 60)
            
            # Load data from JSON if not already loaded
            if self.data_loader.matches_data.empty:
                print("Loading data from JSON...")
                self.data_loader.load_from_json(filepath)
            
            # Initialize network analyzer with the loaded data
            self.network_analyzer = NetworkAnalyzer(self.data_loader)
            
            print("Starting RQ1: Contextual Network Analysis")
            print("=" * 50)
            
            print("\n1. Loading data...")
            print("Loading matches and events for context analysis...")
            
            # Step 2: Extract context windows
            print("\n2. Extracting context windows...")
            all_context_windows = []
            
            # Use the loaded matches data correctly
            if hasattr(self.data_loader, 'matches') and self.data_loader.matches:
                matches_to_process = self.data_loader.matches[:max_matches]
                print(f"Processing {len(matches_to_process)} matches...")
                
                for match in matches_to_process:
                    match_id = str(match['match_id'])  # Ensure string format
                    if match_id in self.data_loader.events:
                        print(f"Processing match {match_id} - {match['home_team']} vs {match['away_team']}")
                        windows = self.context_analyzer.extract_context_windows(
                                    events=self.data_loader.events[match_id],
                                    match_id=match_id,
                                    home_team=match['home_team'],  
                                    away_team=match['away_team'] 
                                )
                        all_context_windows.extend(windows)
                    else:
                        print(f"No events found for match {match_id}")
            else:
                print("No matches data available")
                # Try using matches_data DataFrame instead
                if not self.data_loader.matches_data.empty:
                    print("Using matches_data DataFrame...")
                    matches_df = self.data_loader.matches_data.head(max_matches)
                    
                    for _, match_row in matches_df.iterrows():
                        match_id = str(match_row['match_id'])
                        if match_id in self.data_loader.events:
                            print(f"Processing match {match_id}...")
                            windows = self.context_analyzer.extract_context_windows(
                                self.data_loader.events[match_id], match_id
                            )
                            all_context_windows.extend(windows)
            
            print(f"Extracted {len(all_context_windows)} context windows")
            
            # Step 3: Build networks
            print("\n3. Building passing networks...")
            network_data = self.network_builder.build_networks_from_windows(all_context_windows)
            print(f"Built {len(network_data)} networks")
            
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
            print(f"Calculated metrics for {len(results_df)} windows")
            
            # Step 5: Statistical analysis
            print("\n5. Performing statistical analysis...")
            statistical_results = self.statistical_comparator.compare_contexts(results_df)
            
            # Step 6: Generate report
            print("\n6. Generating report...")
            report = self.statistical_comparator.generate_statistical_report()
            
            # Store results
            self.results = {
                'context_windows': all_context_windows,
                'network_metrics': results_df,
                'statistical_results': statistical_results,
                'report': report
            }

            # Save results
            if save_results:
                self._save_results()
            
            print("\nAnalysis complete!")
            print(f"Total context windows analyzed: {len(all_context_windows)}")
            print(f"Total networks built: {len(network_data)}")
            print(f"Final dataset size: {len(results_df)} observations")
            
            # Print summary only if we have data
            if len(results_df) > 0:
                self.print_summary()
            else:
                print("No data to summarize - check data loading and context extraction")
            
            return self.results

    def _save_results(self):
        """Save analysis results"""
        # use the centralized results_dir
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        self.results['network_metrics'].to_csv(output_dir / "rq1_network_metrics.csv", index=False)

        # Save statistical results
        with open(output_dir / "rq1_statistical_results.json", 'w') as f:
            json.dump(self.results['statistical_results'], f, indent=2, default=str)

        # Save report
        with open(output_dir / "rq1_analysis_report.txt", 'w') as f:
            f.write(self.results['report'])

        print(f"Results saved to {output_dir}/")

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
        """Run RQ2: Rule-Based Tactical Recommendations with Comprehensive Metrics"""

        if not self.results or 'network_metrics' not in self.results:
            raise ValueError("RQ1 results not available. Run RQ1 analysis first.")

        print("\n" + "=" * 60)
        print("RUNNING RQ2: RULE-BASED TACTICAL RECOMMENDATIONS")
        print("=" * 60)

        # Initialize recommendation system
        recommender = TacticalRecommender(self.results)
        recommender.initialize_system(self.results['network_metrics'])

        # Analyze recommendations for all matches
        print("\nGenerating match-level recommendations...")
        match_recommendations = self._generate_match_recommendations(recommender)

        # Calculate comprehensive RQ2 metrics
        print("\nCalculating recommendation system metrics...")
        rq2_metrics = self._calculate_rq2_metrics(recommender, match_recommendations)

        # Create recommendation report
        print("\nCreating recommendation report...")
        recommendation_report = self._create_recommendation_report(
            recommender, match_recommendations, rq2_metrics
        )

        # Store RQ2 results
        rq2_results = {
            'recommender': recommender,
            'match_recommendations': match_recommendations,
            'recommendation_report': recommendation_report,
            'system_summary': recommender.get_system_summary(),
            'rq2_metrics': rq2_metrics  # NEW: Comprehensive metrics
        }

        # Save results
        if save_results:
            self._save_rq2_results(rq2_results)

        # Add to main results
        self.results['rq2_results'] = rq2_results

        print(f"\nRQ2 Analysis Complete!")
        print(f"Generated recommendations for {len(match_recommendations)} match scenarios")
        print(f"Total windows analyzed: {rq2_metrics['coverage_metrics']['total_windows']}")
        print(f"Coverage: {rq2_metrics['coverage_metrics']['coverage_percentage']:.1f}%")

        return rq2_results


    def _calculate_rq2_metrics(self, recommender, match_recommendations: list) -> dict:
        """
        Calculate comprehensive RQ2 metrics as specified in thesis report.
        
        Sections:
        - 4.2.3.1 Rule Activation Metrics
        - 4.2.3.2 Confidence Scoring
        - 4.2.3.3 Temporal Consistency
        """
        
        # Collect all window recommendations across matches
        all_windows = []
        for match in match_recommendations:
            all_windows.extend(match['window_recommendations'])
        
        # ========================================================================
        # 4.2.3.1 RULE ACTIVATION METRICS
        # ========================================================================
        
        rule_activation_metrics = self._calculate_rule_activation_metrics(
            recommender, all_windows
        )
        
        # ========================================================================
        # 4.2.3.2 CONFIDENCE SCORING METRICS
        # ========================================================================
        
        confidence_metrics = self._calculate_confidence_metrics(all_windows)
        
        # ========================================================================
        # 4.2.3.3 TEMPORAL CONSISTENCY METRICS
        # ========================================================================
        
        temporal_metrics = self._calculate_temporal_consistency_metrics(
            match_recommendations
        )
        
        # ========================================================================
        # COVERAGE AND DIVERSITY METRICS
        # ========================================================================
        
        coverage_metrics = self._calculate_coverage_metrics(all_windows)
        
        return {
            'rule_activation_metrics': rule_activation_metrics,
            'confidence_metrics': confidence_metrics,
            'temporal_consistency_metrics': temporal_metrics,
            'coverage_metrics': coverage_metrics,
            'total_windows_analyzed': len(all_windows),
            'total_matches_analyzed': len(match_recommendations)
        }


    def _calculate_rule_activation_metrics(self, recommender, all_windows: list) -> dict:
        """
        Calculate rule activation metrics (Section 4.2.3.1).
        
        Metrics:
        - Trigger frequency: Number of times each rule activated
        - Context specificity: Rule activation distribution across contexts
        - Coverage: Proportion of windows receiving recommendations
        - Diversity: Number of unique recommendations generated
        """
        from collections import Counter, defaultdict
        
        # Track rule activations
        rule_triggers = Counter()
        rule_by_context = defaultdict(lambda: defaultdict(int))
        recommendation_types = []
        windows_with_recs = 0
        
        for window in all_windows:
            if window['recommendations']:
                windows_with_recs += 1
                
                for rec in window['recommendations']:
                    # Count recommendation types (proxy for rule activation)
                    rec_type = rec['type']
                    recommendation_types.append(rec_type)
                    rule_triggers[rec_type] += 1
                    
                    # Track context specificity
                    context = window['current_context']
                    score_ctx = context.get('score_context', 'unknown')
                    phase_ctx = context.get('phase_context', 'unknown')
                    intensity_ctx = context.get('intensity_context', 'unknown')
                    
                    context_key = f"{score_ctx}_{phase_ctx}_{intensity_ctx}"
                    rule_by_context[rec_type][context_key] += 1
        
        # Calculate diversity (unique recommendation types)
        unique_recommendations = len(set(recommendation_types))
        
        # Calculate context specificity scores
        context_specificity = {}
        for rec_type, contexts in rule_by_context.items():
            total_activations = sum(contexts.values())
            # Context specificity: entropy-based measure of distribution
            probs = [count / total_activations for count in contexts.values()]
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
            
            # Normalize entropy (max entropy = log2(num_contexts))
            max_entropy = np.log2(len(contexts)) if len(contexts) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            context_specificity[rec_type] = {
                'specificity_score': round(1 - normalized_entropy, 3),  # 1 = very specific, 0 = uniform
                'contexts_activated': len(contexts),
                'total_activations': total_activations,
                'context_distribution': dict(contexts)
            }
        
        return {
            'trigger_frequency': dict(rule_triggers.most_common()),
            'total_triggers': sum(rule_triggers.values()),
            'context_specificity': context_specificity,
            'diversity': {
                'unique_recommendation_types': unique_recommendations,
                'total_recommendations': len(recommendation_types),
                'diversity_ratio': round(unique_recommendations / len(recommendation_types), 3) if recommendation_types else 0
            },
            'coverage': {
                'windows_with_recommendations': windows_with_recs,
                'total_windows': len(all_windows),
                'coverage_proportion': round(windows_with_recs / len(all_windows), 3) if all_windows else 0
            }
        }


    def _calculate_confidence_metrics(self, all_windows: list) -> dict:
        """
        Calculate confidence scoring metrics (Section 4.2.3.2).
        
        Analyzes:
        - Confidence score distribution
        - Component contributions (urgency, context, temporal)
        - Filtering effectiveness
        """
        confidence_scores = []
        urgency_levels = []
        context_specificities = []
        temporal_consistencies = []
        filtered_count = 0
        
        for window in all_windows:
            if window['recommendations']:
                for rec in window['recommendations']:
                    conf_score = rec['confidence_score']
                    confidence_scores.append(conf_score)
                    context_specificities.append(rec['context_specificity'])
                    
                    # Track urgency
                    urgency_levels.append(window['situation_analysis']['urgency_level'])
                
                # Temporal consistency (window-level)
                temporal_consistencies.append(window['temporal_consistency'])
            else:
                # Count windows with no recommendations (filtered out)
                filtered_count += 1
        
        # Calculate statistics
        if confidence_scores:
            conf_array = np.array(confidence_scores)
            
            confidence_distribution = {
                'mean': round(np.mean(conf_array), 3),
                'median': round(np.median(conf_array), 3),
                'std': round(np.std(conf_array), 3),
                'min': round(np.min(conf_array), 3),
                'max': round(np.max(conf_array), 3),
                'quartiles': {
                    'q25': round(np.percentile(conf_array, 25), 3),
                    'q50': round(np.percentile(conf_array, 50), 3),
                    'q75': round(np.percentile(conf_array, 75), 3)
                }
            }
            
            # Confidence by level
            confidence_levels = {
                'low': sum(1 for c in confidence_scores if c < 0.4),
                'medium': sum(1 for c in confidence_scores if 0.4 <= c < 0.6),
                'high': sum(1 for c in confidence_scores if 0.6 <= c < 0.8),
                'very_high': sum(1 for c in confidence_scores if c >= 0.8)
            }
            
            # Component analysis
            component_contributions = {
                'urgency_factor': {
                    'very_high': urgency_levels.count('very_high'),
                    'high': urgency_levels.count('high'),
                    'medium': urgency_levels.count('medium'),
                    'normal': urgency_levels.count('normal')
                },
                'context_weight': {
                    'mean': round(np.mean(context_specificities), 3),
                    'std': round(np.std(context_specificities), 3)
                },
                'temporal_consistency': {
                    'mean': round(np.mean(temporal_consistencies), 3),
                    'std': round(np.std(temporal_consistencies), 3)
                }
            }
        else:
            confidence_distribution = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0,
                'quartiles': {'q25': 0, 'q50': 0, 'q75': 0}
            }
            confidence_levels = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
            component_contributions = {
                'urgency_factor': {'very_high': 0, 'high': 0, 'medium': 0, 'normal': 0},
                'context_weight': {'mean': 0, 'std': 0},
                'temporal_consistency': {'mean': 0, 'std': 0}
            }
        
        return {
            'confidence_distribution': confidence_distribution,
            'confidence_levels': confidence_levels,
            'component_contributions': component_contributions,
            'filtering_effectiveness': {
                'total_recommendations': len(confidence_scores),
                'filtered_windows': filtered_count,
                'filter_rate': round(filtered_count / len(all_windows), 3) if all_windows else 0
            },
            'formula': 'Confidence = base_confidence + context_effect + temporal_boost'
        }


    def _calculate_temporal_consistency_metrics(self, match_recommendations: list) -> dict:
        """
        Calculate temporal consistency metrics (Section 4.2.3.3).
        
        Definition: Stability of recommendations across sliding windows
        Measurement: Proportion of adjacent windows with same recommendation
        """
        all_consistency_scores = []
        match_level_consistency = []
        
        for match in match_recommendations:
            windows = match['window_recommendations']
            
            if len(windows) < 2:
                continue
            
            # Extract primary recommendations for each window
            primary_recs = []
            for window in windows:
                if window['recommendations']:
                    primary_recs.append(window['summary']['primary_focus'])
                else:
                    primary_recs.append(None)
            
            # Calculate adjacent window consistency
            consistent_transitions = 0
            total_transitions = 0
            
            for i in range(len(primary_recs) - 1):
                if primary_recs[i] is not None and primary_recs[i+1] is not None:
                    total_transitions += 1
                    if primary_recs[i] == primary_recs[i+1]:
                        consistent_transitions += 1
            
            if total_transitions > 0:
                match_consistency = consistent_transitions / total_transitions
                match_level_consistency.append(match_consistency)
                
                # Also collect window-level consistency scores
                for window in windows:
                    all_consistency_scores.append(window['temporal_consistency'])
        
        # Calculate overall statistics
        if match_level_consistency:
            return {
                'overall_consistency': {
                    'mean': round(np.mean(match_level_consistency), 3),
                    'median': round(np.median(match_level_consistency), 3),
                    'std': round(np.std(match_level_consistency), 3),
                    'min': round(np.min(match_level_consistency), 3),
                    'max': round(np.max(match_level_consistency), 3)
                },
                'window_level_consistency': {
                    'mean': round(np.mean(all_consistency_scores), 3),
                    'std': round(np.std(all_consistency_scores), 3)
                },
                'interpretation': {
                    'high_consistency_threshold': 0.7,
                    'matches_with_high_consistency': sum(1 for c in match_level_consistency if c > 0.7),
                    'matches_with_low_consistency': sum(1 for c in match_level_consistency if c < 0.3)
                },
                'measurement': 'Proportion of adjacent windows with same primary recommendation'
            }
        else:
            return {
                'overall_consistency': {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0},
                'window_level_consistency': {'mean': 0, 'std': 0},
                'interpretation': {
                    'high_consistency_threshold': 0.7,
                    'matches_with_high_consistency': 0,
                    'matches_with_low_consistency': 0
                },
                'measurement': 'Proportion of adjacent windows with same primary recommendation'
            }


    def _calculate_coverage_metrics(self, all_windows: list) -> dict:
        """Calculate coverage metrics for recommendation system."""
        windows_with_recs = sum(1 for w in all_windows if w['recommendations'])
        total_recs = sum(len(w['recommendations']) for w in all_windows)
        
        return {
            'total_windows': len(all_windows),
            'windows_with_recommendations': windows_with_recs,
            'windows_without_recommendations': len(all_windows) - windows_with_recs,
            'coverage_percentage': round(windows_with_recs / len(all_windows) * 100, 1) if all_windows else 0,
            'total_recommendations': total_recs,
            'avg_recommendations_per_window': round(total_recs / len(all_windows), 2) if all_windows else 0
        }


    def _create_recommendation_report(self, recommender, match_recs, rq2_metrics) -> str:
        """Create comprehensive recommendation report with RQ2 metrics."""
        
        report_lines = [
            "=" * 70,
            "TACTICAL RECOMMENDATION SYSTEM REPORT",
            "=" * 70,
            "",
            "SYSTEM OVERVIEW:",
            f"- Total Rules: {len(recommender.rule_engine.rules)}",
            f"- Threshold Metrics: {len(recommender.threshold_analyzer.thresholds)}",
            f"- Match Analyses: {len(match_recs)}",
            f"- Total Windows: {rq2_metrics['total_windows_analyzed']}",
            "",
            "=" * 70,
            "4.2.3.1 RULE ACTIVATION METRICS",
            "=" * 70,
            ""
        ]
        
        # Rule activation metrics
        activation = rq2_metrics['rule_activation_metrics']
        
        report_lines.extend([
            "TRIGGER FREQUENCY:",
            f"- Total rule activations: {activation['total_triggers']}"
        ])
        
        for rule_type, count in activation['trigger_frequency'].items():
            pct = count / activation['total_triggers'] * 100 if activation['total_triggers'] > 0 else 0
            report_lines.append(f"  * {rule_type}: {count} ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "CONTEXT SPECIFICITY:",
            "(Higher scores indicate rules are more context-specific)"
        ])
        
        for rule_type, spec_data in activation['context_specificity'].items():
            report_lines.append(
                f"  * {rule_type}: {spec_data['specificity_score']} "
                f"({spec_data['contexts_activated']} contexts)"
            )
        
        report_lines.extend([
            "",
            "COVERAGE:",
            f"- Windows with recommendations: {activation['coverage']['windows_with_recommendations']}/{activation['coverage']['total_windows']}",
            f"- Coverage proportion: {activation['coverage']['coverage_proportion']:.3f}",
            "",
            "DIVERSITY:",
            f"- Unique recommendation types: {activation['diversity']['unique_recommendation_types']}",
            f"- Total recommendations: {activation['diversity']['total_recommendations']}",
            f"- Diversity ratio: {activation['diversity']['diversity_ratio']:.3f}",
            "",
            "=" * 70,
            "4.2.3.2 CONFIDENCE SCORING",
            "=" * 70,
            "",
            "FORMULA:",
            "Confidence = base_confidence + context_effect + temporal_boost",
            "",
            "COMPONENTS:"
        ])
        
        # Confidence metrics
        confidence = rq2_metrics['confidence_metrics']
        
        report_lines.extend([
            "  * Urgency Factor Distribution:"
        ])
        for level, count in confidence['component_contributions']['urgency_factor'].items():
            report_lines.append(f"    - {level}: {count}")
        
        report_lines.extend([
            f"  * Context Weight (mean): {confidence['component_contributions']['context_weight']['mean']}",
            f"  * Temporal Consistency (mean): {confidence['component_contributions']['temporal_consistency']['mean']}",
            "",
            "CONFIDENCE DISTRIBUTION:",
            f"- Mean: {confidence['confidence_distribution']['mean']:.3f}",
            f"- Median: {confidence['confidence_distribution']['median']:.3f}",
            f"- Std: {confidence['confidence_distribution']['std']:.3f}",
            f"- Range: [{confidence['confidence_distribution']['min']:.3f}, {confidence['confidence_distribution']['max']:.3f}]",
            "",
            "CONFIDENCE LEVELS:",
            f"- Very High (≥0.8): {confidence['confidence_levels']['very_high']}",
            f"- High (0.6-0.8): {confidence['confidence_levels']['high']}",
            f"- Medium (0.4-0.6): {confidence['confidence_levels']['medium']}",
            f"- Low (<0.4): {confidence['confidence_levels']['low']}",
            "",
            "FILTERING EFFECTIVENESS:",
            f"- Total recommendations: {confidence['filtering_effectiveness']['total_recommendations']}",
            f"- Filtered windows: {confidence['filtering_effectiveness']['filtered_windows']}",
            f"- Filter rate: {confidence['filtering_effectiveness']['filter_rate']:.3f}",
            "",
            "=" * 70,
            "4.2.3.3 TEMPORAL CONSISTENCY",
            "=" * 70,
            "",
            "DEFINITION:",
            "Stability of recommendations across sliding windows",
            "",
            "MEASUREMENT:",
            confidence['temporal_consistency_metrics']['measurement'] if 'temporal_consistency_metrics' in confidence else rq2_metrics['temporal_consistency_metrics']['measurement'],
            ""
        ])
        
        # Temporal consistency metrics
        temporal = rq2_metrics['temporal_consistency_metrics']
        
        report_lines.extend([
            "OVERALL CONSISTENCY:",
            f"- Mean: {temporal['overall_consistency']['mean']:.3f}",
            f"- Median: {temporal['overall_consistency']['median']:.3f}",
            f"- Std: {temporal['overall_consistency']['std']:.3f}",
            f"- Range: [{temporal['overall_consistency']['min']:.3f}, {temporal['overall_consistency']['max']:.3f}]",
            "",
            "INTERPRETATION:",
            f"- High consistency threshold: {temporal['interpretation']['high_consistency_threshold']}",
            f"- Matches with high consistency: {temporal['interpretation']['matches_with_high_consistency']}",
            f"- Matches with low consistency: {temporal['interpretation']['matches_with_low_consistency']}",
            "",
            "=" * 70,
            "MATCH ANALYSIS SUMMARY",
            "=" * 70,
            ""
        ])
        
        # Add match analysis summary
        if match_recs:
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
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])
        
        return "\n".join(report_lines)


    def _save_rq2_results(self, rq2_results: dict):
        """Save RQ2 results to files including comprehensive metrics."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Save recommendation report
        report_path = self.results_dir / "rq2_recommendation_report.txt"
        with open(report_path, 'w') as f:
            f.write(rq2_results['recommendation_report'])

        # Save system summary
        summary_path = self.results_dir / "rq2_system_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(rq2_results['system_summary'], f, indent=2)
        
        # Save comprehensive RQ2 metrics
        metrics_path = self.results_dir / "rq2_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(rq2_results['rq2_metrics'], f, indent=2)

        print(f"RQ2 results saved to {self.results_dir}")
        print(f"  - Report: {report_path}")
        print(f"  - Summary: {summary_path}")
        print(f"  - Metrics: {metrics_path}")

    def run_rq3_analysis(self, save_results: bool = True) -> Dict:
        """Run RQ3: Recommendation Validation"""
        
        if not self.results or 'rq2_results' not in self.results:
            raise ValueError("RQ2 results not available. Run RQ2 analysis first.")
        
        print("\n" + "="*60)
        print("RUNNING RQ3: RECOMMENDATION VALIDATION")
        print("="*60)
        
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
        validation_results = validator.run_recommendation_validation()
        
        # 2. Run counterfactual analysis
        print("\n2. Running counterfactual analysis...")
        counterfactual_analyzer = CounterfactualAnalyzer(
            self.results['network_metrics'],
            recommendations_data
        )
        counterfactual_results = counterfactual_analyzer.run_counterfactual_analysis()
        
        # 3. Create validation report
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
        
        print(f"\nRQ3 Analysis Complete!")
        print(f"Validation Score: {validation_results['overall_validation_score']['overall_validation_score']:.3f}")
        
        return rq3_results

    def _create_rq3_report(self, validation_results: Dict, 
                        counterfactual_results: Dict) -> str:
        """Create RQ3 validation report"""
        
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
            report_lines.append("STRONG VALIDATION: Recommendations show high effectiveness")
        elif overall_score >= 0.6:
            report_lines.append("MODERATE VALIDATION: Recommendations show good effectiveness")
        else:
            report_lines.append("WEAK VALIDATION: Recommendations need improvement")
        
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
                report_lines.append("High likelihood of performance improvement")
            elif improvement_rate > 0.4:
                report_lines.append("Moderate likelihood of performance improvement")
            else:
                report_lines.append("Low likelihood of performance improvement")
        
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
                report_lines.append("High-quality predictive models")
            elif overall_quality > 0.5:
                report_lines.append("Moderate-quality predictive models")
            else:
                report_lines.append("Low-quality predictive models")

            # Add per-metric RMSE, MAE, R²
            if 'individual_models' in model_perf:
                report_lines.append("")
                report_lines.append("   Model Metrics (per network metric):")
                report_lines.append("   Metric                |   R²    |   MAE   |  RMSE")
                report_lines.append("   ---------------------|---------|---------|---------")
                for metric, perf in model_perf['individual_models'].items():
                    r2 = perf.get('r2', float('nan'))
                    mae = perf.get('mae', float('nan'))
                    rmse = perf.get('rmse', float('nan'))
                    report_lines.append(f"   {metric:<21} | {r2:7.3f} | {mae:7.3f} | {rmse:7.3f}")

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
        
        print(f"RQ3 results saved to {self.results_dir}")

if __name__ == "__main__":
    pass
