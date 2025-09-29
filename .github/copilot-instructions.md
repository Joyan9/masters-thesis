# Copilot Instructions for AI Coding Agents

## Project Overview
This repository analyzes football (soccer) match data using a modular Python codebase. The main focus is on tactical, network, and performance analysis using StatsBomb event data, with outputs for research questions (RQ1, RQ2, RQ3) and coaching insights.

## Architecture & Key Components
- **tactical_analysis/**: Core analysis modules (motif, network, performance, simulation, context classification). Each file is a focused analysis or utility (e.g., `network_analyzer.py`, `motif_analyzer.py`).
- **tactical_analysis_system/**: Higher-level orchestration, system-level analysis, and integration (e.g., `main_analysis.py`, `run_rq2.py`, `tactical_recommender.py`).
- **adhoc_scripts/**: One-off scripts and notebooks for plotting, debugging, and data exploration.
- **analysis_output/**, **results/**: Generated outputs, plots, and reports for each research question.
- **data/**: Raw and interim data files (e.g., StatsBomb JSONs).

## Data Flow
- Data is loaded from `data/` or interim JSONs (e.g., `statsbomb_data_interim_100.json`).
- Analysis modules process data and output results to `analysis_output/` or `results/`.
- Plots and reports are saved in subfolders (e.g., `plots/`, `.txt`, `.json`).

## Developer Workflows
- **Run main analyses:**
  - `python tactical_analysis/main.py` (core analysis)
  - `python tactical_analysis_system/main_analysis.py` (system-level)
  - For RQ2/RQ3: `python tactical_analysis_system/run_rq2.py`, `run_rq3.py`
- **Generate plots:** Use scripts in `adhoc_scripts/` (e.g., `plotting_notebook.ipynb`).
- **Outputs:** Check `analysis_output/` and `results/` for generated files.

## Conventions & Patterns
- **Zone System:** 7x7 grid (49 zones) for pitch analysis. See `plot_7x7_grid` in `plotting_notebook.ipynb`.
- **Adjacency Matrices:** Used for passing networks, see `plot_adjacency_matrix_from_real_data`.
- **Data Orientation:** For period 2, pitch coordinates are flipped (see zone extraction logic).
- **File Naming:** Outputs are named by analysis type and match ID (e.g., `adjacency_matrix_<match_id>.png`).
- **No central config:** Paths and parameters are often hardcoded in scripts.

## Integration & Dependencies
- **Python packages:** `matplotlib`, `networkx`, `seaborn`, `pandas`, `numpy` (install via pip if missing).
- **No build system:** Scripts are run directly; no Makefile or requirements.txt.
- **No formal tests:** Validation is via output inspection and ad hoc scripts.

## Examples
- To visualize the 7x7 grid: run `plot_7x7_grid(output_dir="plots")` in `plotting_notebook.ipynb`.
- To analyze a match: run `plot_adjacency_matrix_from_real_data(json_file, match_id, team_name)`.

## Tips for AI Agents
- Prefer updating or extending modules in `tactical_analysis/` and `tactical_analysis_system/` for new analyses.
- Follow the 7x7 grid and adjacency matrix conventions for network analysis.
- When adding new outputs, save to `analysis_output/` or `results/` with descriptive filenames.
- Use existing scripts as templates for new workflows.
