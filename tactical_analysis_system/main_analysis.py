import logging
import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from tactical_analysis_system.data_loader import DataLoader
from tactical_analysis_system.context_analyzer import ContextAnalyzer
from tactical_analysis_system.network_builder import NetworkBuilder
from tactical_analysis_system.network_analyzer import NetworkAnalyzer
from tactical_analysis_system.statistical_comparator import StatisticalComparator

# ---------- Configuration & Logging ----------
# Ensure package modules importable when running script directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SAVE_DIR = Path("analysis_output")
DEFAULT_SAVE_DIR.mkdir(exist_ok=True)
PLOTS_DIR = DEFAULT_SAVE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("tactical_analysis")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
LOG.addHandler(handler)

# ---------- Helpers ----------

def print_header(title: str) -> None:
    LOG.info("=" * 60)
    LOG.info(f"=== {title} ===")
    LOG.info("=" * 60)

def save_json_safe(data: Any, filepath: Path) -> None:
    """Save data to JSON with proper serialization handling"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def json_serializer(obj):
        """Custom JSON serializer for numpy/pandas objects"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    try:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=json_serializer)
        LOG.info(f"Saved: {filepath}")
    except Exception as e:
        LOG.error(f"Failed to save JSON to {filepath}: {e}")
        # Try saving a simplified version
        try:
            simplified_data = {"error": f"Could not serialize full data: {str(e)}", "keys": list(data.keys()) if isinstance(data, dict) else "non-dict data"}
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            LOG.warning(f"Saved simplified version to {filepath}")
        except:
            LOG.error(f"Could not save even simplified version to {filepath}")

def parse_competitions(comp_str: Optional[str]) -> List[Tuple[int, int]]:
    """Parse competition string like "43,3 11,90" into list of (comp_id, season_id)."""
    if not comp_str:
        return [
            # La Liga
            (11, 90),   # La Liga 2020/2021
            (11, 42),   # La Liga 2019/2020
            # Premier League
            (2, 27),    # Premier League 2015/2016
            # Serie A
            (12, 27),   # Serie A 2015/2016
        ]
    comps = []
    for token in comp_str.split():
        try:
            a, b = token.split(",")
            comps.append((int(a.strip()), int(b.strip())))
        except Exception:
            LOG.warning(f"Ignoring invalid competition token: {token}")
            continue
    return comps if comps else parse_competitions(None)

def save_current_plot(name: str, close: bool = True) -> Path:
    """Save the current matplotlib figure to analysis_output/plots/<name>.png"""
    filename = f"{name.replace(' ', '_').lower()}.png"
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    LOG.info(f"Plot saved: {save_path}")
    if close:
        plt.close()
    return save_path

# ---------- Pipeline Stage Functions ----------

def stage_data_loading(competitions: List[Tuple[int, int]], max_matches: int = 10,
                       save_file: Optional[Path] = DEFAULT_SAVE_DIR / "loaded_data.json") -> DataLoader:
    print_header("DATA LOADING")
    dl = DataLoader()
    LOG.info(f"Loading competitions: {competitions} (max_matches={max_matches})")
    dl.load_data(competitions, max_matches=max_matches)
    if save_file:
        try:
            dl.save_data(str(save_file))
            LOG.info(f"Data saved to {save_file}")
        except Exception as e:
            LOG.exception(f"Failed to save data: {e}")
    return dl

def stage_contextual_analysis(data_loader: DataLoader) -> tuple:
    print_header("CONTEXTUAL NETWORK ANALYSIS (RQ1)")
    
    # Initialize analyzers
    context_analyzer = ContextAnalyzer()
    network_builder = NetworkBuilder()
    network_analyzer = NetworkAnalyzer()
    comparator = StatisticalComparator()
    
    all_results = []
    processed_matches = 0
    
    # Process each match
    for match in data_loader.matches:
        match_id = match["match_id"]
        
        # Check if we have events for this match
        if match_id not in data_loader.events:
            LOG.warning(f"No events found for match {match_id}, skipping...")
            continue
            
        events = data_loader.events[match_id]
        LOG.info(f"Processing match {match_id} ({len(events)} events)...")
        
        try:
            # Extract contexts
            contexts = context_analyzer.extract_match_contexts(events, match_id)
            LOG.info(f"  Extracted contexts: {list(contexts.keys())}")
            
            # Build networks
            networks = network_builder.build_contextual_networks(events, contexts, match_id)
            LOG.info(f"  Built networks for {len(networks)} context types")
            
            # Analyze networks
            results = network_analyzer.analyze_contextual_networks(networks, match_id)
            all_results.append(results)
            processed_matches += 1
            LOG.info(f"  Analyzed {len(results)} context periods")
            
        except Exception as e:
            LOG.error(f"Error processing match {match_id}: {e}")
            continue
    
    LOG.info(f"Successfully processed {processed_matches} matches")
    
    if processed_matches == 0:
        LOG.error("No matches were successfully processed!")
        return pd.DataFrame(), {}, "No analysis could be performed - no matches processed successfully."
    
    # Combine all results
    combined_results = network_analyzer.get_aggregated_results()
    LOG.info(f"Combined results: {len(combined_results)} total context periods")
    
    if len(combined_results) == 0:
        LOG.error("No results generated from processed matches!")
        return pd.DataFrame(), {}, "No analysis results generated."
    
    # Statistical comparison
    comparisons = comparator.compare_contexts(combined_results)
    LOG.info(f"Completed statistical comparisons for {len(comparisons)} context types")
    
    # Generate and save report
    report = comparator.generate_summary_report()
    
    # Save results
    results_file = DEFAULT_SAVE_DIR / 'rq1_network_metrics.csv'
    combined_results.to_csv(results_file, index=False)
    LOG.info(f"Results saved to {results_file}")
    
    # Save report
    report_file = DEFAULT_SAVE_DIR / 'rq1_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    LOG.info(f"Report saved to {report_file}")
    
    # Save detailed comparisons (now JSON serializable)
    comparisons_file = DEFAULT_SAVE_DIR / 'rq1_statistical_comparisons.json'
    save_json_safe(comparisons, comparisons_file)
    
    # Save summary statistics
    summary_stats = comparator.get_summary_statistics()
    summary_file = DEFAULT_SAVE_DIR / 'rq1_summary_statistics.json'
    save_json_safe(summary_stats, summary_file)
    
    return combined_results, comparisons, report

def run_rq1_analysis(competitions_str: Optional[str] = None, max_matches: int = 10):
    """Complete pipeline for RQ1 analysis"""
    
    # Parse competitions
    competitions = parse_competitions(competitions_str)
    
    # Stage 1: Data Loading
    data_loader = stage_data_loading(competitions, max_matches)
    
    if not data_loader.events:
        LOG.error("No events data loaded. Cannot proceed with analysis.")
        return None, None, None
    
    # Stage 2: Contextual Analysis
    results, comparisons, report = stage_contextual_analysis(data_loader)
    
    # Print summary
    print_header("ANALYSIS COMPLETE")
    LOG.info(f"Processed {len(data_loader.matches)} matches")
    LOG.info(f"Generated {len(results)} context period analyses")
    LOG.info("Check analysis_output/ directory for detailed results")
    
    print("\n" + report)
    
    return results, comparisons, report

if __name__ == "__main__":
    # Example usage
    results, comparisons, report = run_rq1_analysis(
        competitions_str="11,90 2,27",  # La Liga 2020/21 + Premier League 2015/16
        max_matches=5
    )
