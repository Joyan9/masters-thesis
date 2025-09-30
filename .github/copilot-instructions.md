# Copilot Instructions for AI Coding Agents

## Project Overview
This repository analyzes football (soccer) match data using a modular Python codebase. The main focus is on tactical, network, and performance analysis using StatsBomb event data, with outputs for research questions (RQ1, RQ2, RQ3) and coaching insights.
- RQ1: Contextual Network Analysis How do passing network characteristics (density, centrality measures, clustering coefficient) differ between distinct match contexts (winning vs. losing vs. tied game states) within individual football matches?
- RQ2: Rule-Based Tactical Recommendations Can we develop a rule-based system that translates observed network patterns into actionable tactical suggestions (e.g., "increase wing play when central density drops below threshold X")?
- RQ3: Recommendation Validation Can we validate the effectiveness of our rule-based tactical recommendations through counterfactual simulation analysis of historical match scenarios?


## Architecture & Key Components
- **tactical_analysis_system/**: Main orchestration and analysis modules (e.g., `main_analysis.py`, `run_rq2.py`, `run_rq3.py`, `tactical_recommender.py`, `visualizer.py`). Each file is focused on a specific analysis, workflow, or utility.
- **adhoc_scripts/**: One-off scripts and notebooks for plotting, debugging, and data exploration.
- **results/**: Generated outputs, plots, and reports for each research question.

## Data Flow
- Data is loaded from JSONs (e.g., `statsbomb_data_interim_100.json`).
- Analysis modules process data and output results to `results/`.
- Plots and reports are saved in subfolders (e.g., `plots/`, `.txt`, `.json`).

## Developer Workflows
- **Run main analyses:**
  - `python tactical_analysis_system/main_analysis.py` (system-level)
  - For RQ2/RQ3: `python tactical_analysis_system/run_rq2.py`, `python tactical_analysis_system/run_rq3.py`
- **Generate plots:** Use scripts in `adhoc_scripts/` (e.g., `plotting_notebook.ipynb`).
- **Outputs:** Check `results/` for generated files.

## Conventions & Patterns
- **Zone System:** 7x7 grid (49 zones) for pitch analysis. See `plot_7x7_grid` in `plotting_notebook.ipynb`.
- **Adjacency Matrices:** Used for passing networks, see `plot_adjacency_matrix_from_real_data`.
- **Data Orientation:** For period 2, pitch coordinates are flipped (see zone extraction logic).
- **File Naming:** Outputs are named by analysis type and match ID (e.g., `adjacency_matrix_<match_id>.png`).
- **No central config:** Paths and parameters are often hardcoded in scripts.
- **Sliding Window Analysis:** Use or extend logic in `run_rq3.py` for dynamic network analysis with customizable window and step sizes.
- **Tactical Recommendation Rules:** Tactical rules are mapped to network metric thresholds (see `tactical_recommender.py` and `network_metric_rule_map.json`).

## Integration & Dependencies
- **Python packages:** `matplotlib`, `networkx`, `seaborn`, `pandas`, `numpy` (install via pip if missing).
- **No build system:** Scripts are run directly; no Makefile or requirements.txt.
- **No formal tests:** Validation is via output inspection and ad hoc scripts.

## Examples
- To visualize the 7x7 grid: run `plot_7x7_grid(output_dir="plots")` in `plotting_notebook.ipynb`.
- To analyze a match: run `plot_adjacency_matrix_from_real_data(json_file, match_id, team_name)`.
- To build passing networks with different window sizes: see or extend `run_rq3.py`.
- To generate tactical recommendations: use `TacticalRecommender` with network metrics and context.

## Tips for AI Agents
- Prefer updating or extending modules in `tactical_analysis_system/` for new analyses.
- Follow the 7x7 grid and adjacency matrix conventions for network analysis.
- When adding new outputs, save to `results/` with descriptive filenames.
- Use existing scripts as templates for new workflows.
- Reference `tactical_recommender.py` and `tactical_rules.yml` for tactical rule logic and threshold mappings.
