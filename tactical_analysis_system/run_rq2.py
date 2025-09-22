# Complete RQ1 + RQ2 Analysis
from .main_analysis import MainAnalysis

# Initialize
analysis = MainAnalysis()

# Run RQ1 first
rq1_results = analysis.run_full_analysis(
    competitions=[(2, 27)],
    max_matches=5,
    save_results=True,
    create_plots=True
)

# Run RQ2 using RQ1 results
rq2_results = analysis.run_rq2_analysis(save_results=True)

# Test specific scenario
recommender = rq2_results['recommender']

test_metrics = {
    'density': 0.025,  # Low
    'clustering_coefficient': 0.020,
    'centralization': 0.080
}

test_context = {
    'score_context': 'trailing',
    'phase_context': 'late',
    'intensity_context': 'low'
}

recommendations = recommender.get_recommendations(test_metrics, test_context)
print("Tactical Recommendations:")
for rec in recommendations['recommendations']:
    print(f"- {rec['action']} (Confidence: {rec['confidence']})")
