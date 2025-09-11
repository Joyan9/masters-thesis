"""
Main execution script for Tactical Analysis System
Orchestrates the complete analysis pipeline from Days 3-7
"""

import sys
import os
from datetime import datetime

# Add the tactical_analysis package to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tactical_analysis.data_loader import DataLoader
from tactical_analysis.context_classifier import TacticalContextClassifier
from tactical_analysis.network_analyzer import BaselineNetworkAnalyzer
from tactical_analysis.motif_analyzer import MotifAnalyzer
from tactical_analysis.coaching_insights import CoachingInsightsEngine



def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"=== {title} ===")
    print("="*60)

def run_days_3_4(data_loader, max_matches=10):
    """Run Days 3-4: Context Definition & Classification"""
    print_header("DAYS 3-4: CONTEXT DEFINITION & CLASSIFICATION")
    
    # Initialize context classifier
    classifier = TacticalContextClassifier(data_loader)
    
    # Process context classifications
    print("\n=== PROCESSING CONTEXT CLASSIFICATIONS ===")
    classifications = classifier.process_multiple_matches()
    
    # Validate context categories
    print("\n=== VALIDATING CONTEXT CATEGORIES ===")
    classifier.validate_context_categories()
    
    # Visualize distributions
    print("\n=== GENERATING CONTEXT VISUALIZATIONS ===")
    classifier.visualize_context_distributions()
    
    # Show sample context transitions
    print("\n=== SAMPLE CONTEXT TRANSITIONS ===")
    if classifier.context_transitions:
        sample_match = list(classifier.context_transitions.keys())[0]
        sample_transitions = classifier.context_transitions[sample_match]
        
        print(f"Sample match {sample_match} transitions:")
        for team, transitions in sample_transitions.items():
            print(f"\n{team}:")
            for transition in transitions['score_transitions']:
                print(f"  Score: {transition['from']} → {transition['to']} in {transition['phase']} phase")
    
    # Save results
    classifier.save_context_data('tactical_contexts_days3_4.json')
    
    print("\n✅ Days 3-4 Complete!")
    print("✅ Context categories operationalized")
    print("✅ Context transition detection implemented")
    print("✅ Context categories validated with sample matches")
    
    return classifier

def run_days_5_7(classifier):
    """Run Days 5-7: Baseline Network Analysis"""
    print_header("DAYS 5-7: BASELINE NETWORK ANALYSIS")
    
    # Initialize network analyzer
    print("\n=== INITIALIZING NETWORK ANALYZER ===")
    network_analyzer = BaselineNetworkAnalyzer(classifier)
    
    # Process network analysis for all matches
    print("\n=== PROCESSING NETWORK ANALYSIS ===")
    network_analyzer.process_multiple_matches()
    
    # Static analysis: Compare centrality measures across contexts
    print("\n=== STATIC ANALYSIS: CONTEXT COMPARISON ===")
    statistical_results = network_analyzer.compare_contexts_static()
    
    # Dynamic analysis: Track centrality changes in rolling windows
    print("\n=== DYNAMIC ANALYSIS: ROLLING WINDOW PATTERNS ===")
    if network_analyzer.network_metrics:
        sample_match = list(network_analyzer.network_metrics.keys())[0]
        dynamic_patterns = network_analyzer.analyze_dynamic_patterns(sample_match)
        
        if dynamic_patterns:
            print(f"Sample dynamic analysis for match {sample_match}:")
            for team, patterns in dynamic_patterns.items():
                print(f"\n{team}:")
                for trend in patterns['trends'][:3]:  # Show first 3 trends
                    print(f"  {trend['metric']}: {trend['trend_direction']} trend (R²={trend['r_squared']:.3f})")
    
    # Vulnerability detection
    print("\n=== VULNERABILITY DETECTION ===")
    network_analyzer.generate_vulnerability_report()
    
    # Visualizations
    print("\n=== GENERATING NETWORK VISUALIZATIONS ===")
    network_analyzer.visualize_network_analysis()
    
    # Save results
    print("\n=== SAVING NETWORK ANALYSIS RESULTS ===")
    network_analyzer.save_network_analysis('network_analysis_days5_7.json')
    
    print("\n✅ Days 5-7 Complete!")
    print("✅ Network signatures of tactical vulnerability identified")
    print("✅ Static analysis with Mann-Whitney U tests completed")
    print("✅ Dynamic analysis with 10-minute rolling windows completed")
    print("✅ Vulnerability threshold values established")
    print("✅ Statistical tests and effect size calculations performed")
    
    return network_analyzer

def run_days_8_9(network_analyzer):
    """Run Days 8-9: Motif Analysis & Pattern Recognition"""
    print_header("DAYS 8-9: MOTIF ANALYSIS & PATTERN RECOGNITION")
    
    # Initialize motif analyzer
    print("\n=== INITIALIZING MOTIF ANALYZER ===")
    motif_analyzer = MotifAnalyzer(network_analyzer)
    
    # Process motif analysis
    print("\n=== PROCESSING MOTIF ANALYSIS ===")
    motif_analyzer.process_multiple_matches()
    
    # Compare motif patterns across contexts
    print("\n=== COMPARING MOTIF CONTEXTS ===")
    motif_analyzer.compare_motif_contexts()
    
    # Visualize motif patterns
    print("\n=== GENERATING MOTIF VISUALIZATIONS ===")
    motif_analyzer.visualize_motif_patterns()
    
    # Initialize coaching insights engine
    print("\n=== INITIALIZING COACHING INSIGHTS ENGINE ===")
    coaching_engine = CoachingInsightsEngine(network_analyzer, motif_analyzer)
    
    # Generate coaching insights
    print("\n=== GENERATING COACHING INSIGHTS ===")
    coaching_engine.process_multiple_matches()
    
    # Generate sample coaching report
    print("\n=== SAMPLE COACHING REPORT ===")
    if coaching_engine.insights:
        sample_match = list(coaching_engine.insights.keys())[0]
        coaching_engine.generate_coaching_report(sample_match)
    
    # Save results
    print("\n=== SAVING RESULTS ===")
    motif_analyzer.save_motif_analysis()
    coaching_engine.save_coaching_insights()
    
    print("\n✅ Days 8-9 Complete!")
    print("✅ 3-node motif analysis implemented")
    print("✅ Temporal motif patterns identified")
    print("✅ Context-specific pattern analysis completed")
    print("✅ Rule-based coaching insights generated")
    print("✅ Confidence scoring system implemented")
    
    return motif_analyzer, coaching_engine

def run_full_analysis(competitions=None, max_matches=10, save_data=True):
    """Run the complete analysis pipeline (Days 3-7)"""
    print_header("TACTICAL ANALYSIS SYSTEM - FULL PIPELINE")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Default competitions if none provided
    if competitions is None:
        competitions = [
            (43, 3),  # World Cup 2018
            # Add more competitions as needed
        ]
    
    # Step 1: Load Data
    print_header("DATA LOADING")
    data_loader = DataLoader()
    data_loader.load_data(competitions, max_matches=max_matches)
    
    if save_data:
        data_loader.save_data('loaded_data.json')
        print("✅ Data saved for future use")
    
    # Step 2: Run Days 3-4 Analysis
    classifier = run_days_3_4(data_loader, max_matches)
    
    # Step 3: Run Days 5-7 Analysis
    network_analyzer = run_days_5_7(classifier)

    # Step 4: Run Days 8-9 Analysis
    motif_analyzer, coaching_engine = run_days_8_9(network_analyzer)

    # Final Summary
    print_header("ANALYSIS COMPLETE - SUMMARY")
    print(f"📊 Matches analyzed: {len(data_loader.matches)}")
    print(f"📊 Events processed: {len(data_loader.events)}")
    print(f"📊 Context classifications: {len(classifier.context_classifications)}")
    print(f"📊 Network analyses: {len(network_analyzer.network_metrics)}")
    print(f"📊 Vulnerability signatures: {len(network_analyzer.vulnerability_signatures)}")
    print(f"📊 Motif patterns: {len(motif_analyzer.motif_patterns)}")
    print(f"📊 Coaching insights: {len(coaching_engine.insights)}")
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'data_loader': data_loader,
        'classifier': classifier,
        'network_analyzer': network_analyzer,
        'motif_analyzer': motif_analyzer,
        'coaching_engine': coaching_engine
    }
    

def run_from_saved_data(data_file='loaded_data.json'):
    """Run analysis from previously saved data"""
    print_header("RUNNING ANALYSIS FROM SAVED DATA")
    
    # Load saved data
    data_loader = DataLoader()
    try:
        data_loader.load_saved_data(data_file)
        print(f"✅ Loaded data from {data_file}")
    except FileNotFoundError:
        print(f"❌ Could not find {data_file}. Please run full analysis first.")
        return None
    
    # Run analysis pipeline
    classifier = run_days_3_4(data_loader)
    network_analyzer = run_days_5_7(classifier)
    
    return {
        'data_loader': data_loader,
        'classifier': classifier,
        'network_analyzer': network_analyzer
    }

def run_quick_test(max_matches=3):
    """Run a quick test with minimal data for development/testing"""
    print_header("QUICK TEST MODE")
    print("Running analysis with minimal data for testing...")
    
    competitions = [(43, 3)]  # Just World Cup 2018
    return run_full_analysis(competitions, max_matches=max_matches)

def run_custom_analysis():
    """Run custom analysis with user-defined parameters"""
    print_header("CUSTOM ANALYSIS MODE")
    
    # Get user input
    print("Available competitions:")
    print("43, 3 - World Cup 2018")
    print("11, 1 - La Liga 2018/19")
    print("2, 44 - Premier League 2018/19")
    
    comp_input = input("Enter competition IDs (format: comp_id,season_id comp_id,season_id): ")
    max_matches_input = input("Enter max matches per competition (default 10): ")
    
    # Parse input
    try:
        competitions = []
        for comp_str in comp_input.split():
            comp_id, season_id = map(int, comp_str.split(','))
            competitions.append((comp_id, season_id))
        
        max_matches = int(max_matches_input) if max_matches_input else 10
        
    except ValueError:
        print("Invalid input format. Using defaults.")
        competitions = [(43, 3)]
        max_matches = 10
    
    return run_full_analysis(competitions, max_matches)

def main():
    """Main entry point with menu system"""
    print_header("TACTICAL ANALYSIS SYSTEM")
    
    run_full_analysis()
    
if __name__ == "__main__":
    # Run the main menu system
    results = main()
    
    if results:
        print("\n🎉 Analysis completed successfully!")
