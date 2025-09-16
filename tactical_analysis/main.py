#!/usr/bin/env python3
"""
main.py — Refactored orchestration script for Tactical Analysis System

Goals:
- Provide a CLI interface to run the full pipeline or individual stages.
- Cleaner separation of responsibilities and reusable helpers for load/save.
- Logging instead of printing (still prints important status lines).
- Backwards-compatible with the tactical_analysis package API described in your repo.
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

# Ensure package modules importable when running script directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Tactical analysis modules (assumes package structure from your description)
from tactical_analysis.data_loader import DataLoader
from tactical_analysis.context_classifier import TacticalContextClassifier
from tactical_analysis.network_analyzer import BaselineNetworkAnalyzer
from tactical_analysis.motif_analyzer import MotifAnalyzer
from tactical_analysis.coaching_insights import CoachingInsightsEngine
from tactical_analysis.performance_analyzer import PerformanceAnalyzer
from tactical_analysis.simulation_engine import TacticalSimulationEngine
from tactical_analysis.empirical_analysis import EmpiricalPPNAnalyzer


# ---------- Configuration & Logging ----------

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
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    LOG.info(f"Saved: {filepath}")


def load_json_safe(filepath: Path) -> Optional[Dict]:
    try:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        LOG.warning(f"File not found: {filepath}")
        return None


def parse_competitions(comp_str: Optional[str]) -> List[Tuple[int, int]]:
    """
    Parse competition string like "43,3 11,90" into list of (comp_id, season_id).
    Falls back to a sensible default list if None or invalid.
    """
    if not comp_str:
        return [
                # La Liga
                (11, 90),   # La Liga 2020/2021
                (11, 42),   # La Liga 2019/2020
                (11, 4),    # La Liga 2018/2019
                (11, 1),    # La Liga 2017/2018
                (11, 2),    # La Liga 2016/2017
            
                # Ligue 1
                (7, 235),   # Ligue 1 2022/2023
                (7, 108),   # Ligue 1 2021/2022

                # Premier League
                (2, 27),    # Premier League 2015/2016
                (2, 44),    # Premier League 2003/2004

                # Serie A
                (12, 27),   # Serie A 2015/2016
                (12, 86),   # Serie A 1986/1987
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
    """
    Save the current matplotlib figure to analysis_output/plots/<name>.png
    """
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


def stage_context_classification(data_loader: DataLoader,
                                 save_file: Optional[Path] = DEFAULT_SAVE_DIR / "tactical_contexts.json") -> TacticalContextClassifier:
    print_header("CONTEXT CLASSIFICATION (formerly Days 3-4)")
    classifier = TacticalContextClassifier(data_loader)
    LOG.info("Processing context classifications for multiple matches...")
    classifier.process_multiple_matches()
    classifier.validate_context_categories()
    classifier.visualize_context_distributions()
    save_current_plot("context_distributions")
    try:
        classifier.save_context_data(str(save_file))
    except Exception:
        # Fallback: try saving a minimal JSON representation
        try:
            minimal = {
                "context_classifications": classifier.context_classifications,
                "context_transitions": classifier.context_transitions,
                "team_standings": getattr(classifier, "team_standings", {})
            }
            save_json_safe(minimal, save_file)
        except Exception:
            LOG.exception("Failed to save tactical context data.")
    return classifier


def stage_network_analysis(classifier: TacticalContextClassifier,
                           save_file: Optional[Path] = DEFAULT_SAVE_DIR / "network_analysis.json") -> BaselineNetworkAnalyzer:
    print_header("BASELINE NETWORK ANALYSIS (formerly Days 5-7)")
    network_analyzer = BaselineNetworkAnalyzer(classifier)
    network_analyzer.process_multiple_matches()
    stats = network_analyzer.compare_contexts_static()
    if network_analyzer.network_metrics:
        sample_match = next(iter(network_analyzer.network_metrics.keys()))
        patterns = network_analyzer.analyze_dynamic_patterns(sample_match)
        if patterns:
            LOG.info(f"Example dynamic patterns for match {sample_match}: {list(patterns.keys())[:5]}")
    network_analyzer.generate_vulnerability_report()
    network_analyzer.visualize_network_analysis()
    save_current_plot("network_analysis")
    try:
        network_analyzer.save_network_analysis(str(save_file))
    except Exception:
        try:
            minimal = {
                "network_metrics": getattr(network_analyzer, "network_metrics", {}),
                "zone_networks": getattr(network_analyzer, "zone_networks", {}),
                "vulnerability_signatures": getattr(network_analyzer, "vulnerability_signatures", {})
            }
            save_json_safe(minimal, save_file)
        except Exception:
            LOG.exception("Failed to save network analysis data.")
    return network_analyzer


def stage_motif_and_coaching(network_analyzer: BaselineNetworkAnalyzer,
                             save_motif: Optional[Path] = DEFAULT_SAVE_DIR / "motif_analysis.json",
                             save_coaching: Optional[Path] = DEFAULT_SAVE_DIR / "coaching_insights.json"):
    print_header("MOTIF ANALYSIS & COACHING INSIGHTS (formerly Days 8-9)")
    motif_analyzer = MotifAnalyzer(network_analyzer)
    motif_analyzer.process_multiple_matches()
    motif_analyzer.compare_motif_contexts()
    motif_analyzer.visualize_motif_patterns()
    save_current_plot("motif_patterns")

    coaching_engine = CoachingInsightsEngine(network_analyzer, motif_analyzer)
    coaching_engine.process_multiple_matches()
    if coaching_engine.insights:
        sample_match = next(iter(coaching_engine.insights.keys()))
        coaching_engine.generate_coaching_report(sample_match)

    try:
        motif_analyzer.save_motif_analysis(str(save_motif))
    except Exception:
        save_json_safe(getattr(motif_analyzer, "motif_patterns", {}), save_motif)

    try:
        coaching_engine.save_coaching_insights(str(save_coaching))
    except Exception:
        save_json_safe(getattr(coaching_engine, "insights", {}), save_coaching)

    return motif_analyzer, coaching_engine


def stage_performance_and_simulation(network_analyzer: BaselineNetworkAnalyzer,
                                     save_perf: Optional[Path] = DEFAULT_SAVE_DIR / "performance_analysis.json",
                                     save_sim: Optional[Path] = DEFAULT_SAVE_DIR / "simulation_results.json"):
    print_header("PERFORMANCE ANALYSIS & SIMULATION (formerly Days 12-14)")
    perf = PerformanceAnalyzer(network_analyzer)
    perf.process_multiple_matches()
    if perf.goal_analysis:
        sample_match = next(iter(perf.goal_analysis.keys()))
        perf.generate_performance_report(sample_match)
    perf.visualize_performance_analysis()
    save_current_plot("performance_analysis")

    sim = TacticalSimulationEngine(network_analyzer, perf)
    sim.establish_baseline_metrics()
    sim.process_multiple_matches()
    if sim.simulation_results:
        sample_key = next(iter(sim.simulation_results.keys()))
        sim.generate_simulation_report(sim.simulation_results[sample_key])
        sim.visualize_simulation_results(sim.simulation_results[sample_key])
        save_current_plot("simulation_results")

    try:
        perf.save_performance_analysis(str(save_perf))
    except Exception:
        save_json_safe(getattr(perf, "correlation_results", {}), save_perf)

    try:
        sim.save_simulation_results(str(save_sim))
    except Exception:
        save_json_safe(getattr(sim, "simulation_results", {}), save_sim)

    return perf, sim


def stage_empirical_analysis(data_loader: Optional[DataLoader] = None) -> EmpiricalPPNAnalyzer:
    print_header("EMPIRICAL PITCH-PASSING NETWORK ANALYSIS")
    if data_loader is None:
        LOG.info("No DataLoader provided; creating a fresh DataLoader for empirical analysis.")
        data_loader = DataLoader()
        # Use a larger sample for empirical stats (maintain your original default)
        data_loader.load_data([(2, 27)], max_matches=20)
    empirical = EmpiricalPPNAnalyzer(data_loader)
    empirical.run_full_empirical_analysis()
    # Optionally add a save hook if EmpiricalPPNAnalyzer has one
    return empirical


# ---------- Utility: Loading saved context/network ----------

def load_saved_contexts(context_file: Path) -> Optional[TacticalContextClassifier]:
    """
    Loads a TacticalContextClassifier populated from a saved context JSON.
    Returns a classifier instance if load successful, else None.
    """
    raw = load_json_safe(context_file)
    if not raw:
        return None
    dl = DataLoader()  # will have empty matches/events; user should ensure data exists if needed
    classifier = TacticalContextClassifier(dl)
    classifier.context_classifications = raw.get("context_classifications", {})
    classifier.context_transitions = raw.get("context_transitions", {})
    classifier.team_standings = raw.get("team_standings", {})
    LOG.info("Loaded context classifier from saved file.")
    return classifier


def load_saved_network(network_file: Path, classifier: Optional[TacticalContextClassifier] = None) -> Optional[BaselineNetworkAnalyzer]:
    raw = load_json_safe(network_file)
    if not raw:
        return None
    if classifier is None:
        # If no classifier given, create a dummy one
        dl = DataLoader()
        classifier = TacticalContextClassifier(dl)
    network_analyzer = BaselineNetworkAnalyzer(classifier)
    network_analyzer.network_metrics = raw.get("network_metrics", {})
    network_analyzer.zone_networks = raw.get("zone_networks", {})
    network_analyzer.vulnerability_signatures = raw.get("vulnerability_signatures", {})
    LOG.info("Loaded network analyzer from saved file.")
    return network_analyzer


# ---------- CLI Entrypoint ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tactical Analysis System CLI")
    p.add_argument("--mode", "-m", choices=[
        "full", "quick", "from-saved", "custom",
        "context", "network", "motif", "performance", "empirical", "simulation"
    ], default="full", help="Select pipeline mode / stage to run.")
    p.add_argument("--competitions", "-c", type=str,
                   help='Competition list as "comp,season comp,season" (e.g. "43,3 11,90")')
    p.add_argument("--max-matches", "-n", type=int, default=10, help="Max matches per competition.")
    p.add_argument("--save-dir", "-s", type=str, default=str(DEFAULT_SAVE_DIR), help="Directory to save intermediate results.")
    p.add_argument("--context-file", type=str, help="Load context data from JSON file (for network/motif/perf stages).")
    p.add_argument("--network-file", type=str, help="Load network analysis JSON file (for motif/perf stages).")
    return p


def main_cli(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve standard file paths inside save_dir
    context_save = save_dir / "tactical_contexts.json"
    network_save = save_dir / "network_analysis.json"
    motif_save = save_dir / "motif_analysis.json"
    coaching_save = save_dir / "coaching_insights.json"
    perf_save = save_dir / "performance_analysis.json"
    sim_save = save_dir / "simulation_results.json"
    loaded_data_save = save_dir / "loaded_data.json"

    competitions = parse_competitions(args.competitions)

    results: Dict[str, Any] = {}

    try:
        if args.mode == "quick":
            # Minimal test: small dataset, run full pipeline on it
            LOG.info("Running quick test pipeline")
            dl = stage_data_loading([(2, 27)], max_matches=min(3, args.max_matches), save_file=loaded_data_save)
            classifier = stage_context_classification(dl, save_file=context_save)
            network = stage_network_analysis(classifier, save_file=network_save)
            motif, coaching = stage_motif_and_coaching(network, save_motif=motif_save, save_coaching=coaching_save)
            perf, sim = stage_performance_and_simulation(network, save_perf=perf_save, save_sim=sim_save)
            results.update(locals())
            return results

        if args.mode == "from-saved":
            LOG.info("Running analysis from saved data")
            dl = DataLoader()
            try:
                dl.load_saved_data(str(loaded_data_save))
                LOG.info(f"Loaded data from {loaded_data_save}")
            except Exception:
                LOG.warning("No saved loaded data; run data loading first.")
            classifier = None
            if args.context_file:
                classifier = load_saved_contexts(Path(args.context_file))
            else:
                classifier = load_saved_contexts(context_save)
            if not classifier:
                LOG.info("No saved context classifier found; running context classification anew.")
                classifier = stage_context_classification(dl, save_file=context_save)
            network = None
            if args.network_file:
                network = load_saved_network(Path(args.network_file), classifier)
            else:
                network = load_saved_network(network_save, classifier)
            if not network:
                LOG.info("No saved network found; running network analysis anew.")
                network = stage_network_analysis(classifier, save_file=network_save)
            # After loading/creating classifier & network, run motif & perf & sim
            motif, coaching = stage_motif_and_coaching(network, save_motif=motif_save, save_coaching=coaching_save)
            perf, sim = stage_performance_and_simulation(network, save_perf=perf_save, save_sim=sim_save)
            results.update(locals())
            return results

        if args.mode == "full":
            LOG.info("Running full pipeline")
            dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
            classifier = stage_context_classification(dl, save_file=context_save)
            network = stage_network_analysis(classifier, save_file=network_save)
            motif, coaching = stage_motif_and_coaching(network, save_motif=motif_save, save_coaching=coaching_save)
            perf, sim = stage_performance_and_simulation(network, save_perf=perf_save, save_sim=sim_save)
            results.update(locals())
            LOG.info("Full pipeline complete")
            return results

        if args.mode == "custom":
            LOG.info("Running custom analysis (interactive or via --competitions)")
            dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
            classifier = stage_context_classification(dl, save_file=context_save)
            # For custom, don't automatically run all downstream stages — return classifier to user
            results.update({"data_loader": dl, "classifier": classifier})
            return results

        # Stage-level modes
        if args.mode == "context":
            dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
            classifier = stage_context_classification(dl, save_file=context_save)
            results.update({"classifier": classifier, "data_loader": dl})
            return results

        if args.mode == "network":
            # Need classifier (either from file or freshly computed)
            classifier = load_saved_contexts(Path(args.context_file)) if args.context_file else load_saved_contexts(context_save)
            if not classifier:
                dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
                classifier = stage_context_classification(dl, save_file=context_save)
            network = stage_network_analysis(classifier, save_file=network_save)
            results.update({"network_analyzer": network, "classifier": classifier})
            return results

        if args.mode == "motif":
            network = load_saved_network(Path(args.network_file), None) if args.network_file else load_saved_network(network_save, None)
            if not network:
                classifier = load_saved_contexts(context_save)
                if not classifier:
                    dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
                    classifier = stage_context_classification(dl, save_file=context_save)
                network = stage_network_analysis(classifier, save_file=network_save)
            motif, coaching = stage_motif_and_coaching(network, save_motif=motif_save, save_coaching=coaching_save)
            results.update({"motif_analyzer": motif, "coaching_engine": coaching})
            return results

        if args.mode == "performance":
            network = load_saved_network(Path(args.network_file), None) if args.network_file else load_saved_network(network_save, None)
            if not network:
                classifier = load_saved_contexts(context_save)
                if not classifier:
                    dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
                    classifier = stage_context_classification(dl, save_file=context_save)
                network = stage_network_analysis(classifier, save_file=network_save)
            perf, sim = stage_performance_and_simulation(network, save_perf=perf_save, save_sim=sim_save)
            results.update({"performance_analyzer": perf, "simulation_engine": sim})
            return results

        if args.mode == "empirical":
            dl = stage_data_loading(competitions, max_matches=20, save_file=loaded_data_save)
            empirical = stage_empirical_analysis(dl)
            results.update({"empirical_analyzer": empirical})
            return results

        if args.mode == "simulation":
            # Run simulations based on saved network/perf or generate them
            classifier = load_saved_contexts(context_save)
            if not classifier:
                dl = stage_data_loading(competitions, max_matches=args.max_matches, save_file=loaded_data_save)
                classifier = stage_context_classification(dl, save_file=context_save)
            network = load_saved_network(network_save, classifier)
            if not network:
                network = stage_network_analysis(classifier, save_file=network_save)
            perf = PerformanceAnalyzer(network)
            perf.process_multiple_matches()
            sim = TacticalSimulationEngine(network, perf)
            sim.establish_baseline_metrics()
            sim.process_multiple_matches()
            results.update({"simulation_engine": sim})
            return results

        # Fallback
        LOG.warning("Unknown mode — exiting.")
        return {}

    except KeyboardInterrupt:
        LOG.warning("Execution interrupted by user.")
        return results
    except Exception as e:
        LOG.exception(f"Unhandled exception during pipeline: {e}")
        return results


# ---------- Script entry ----------

if __name__ == "__main__":
    print_header("TACTICAL ANALYSIS SYSTEM - START")
    start = datetime.now()
    res = main_cli()
    end = datetime.now()
    print_header("ANALYSIS FINISHED")
    LOG.info(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}, End: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    if res:
        LOG.info("Pipeline produced results; inspect returned objects for details.")
    else:
        LOG.info("No results produced (check logs above).")
