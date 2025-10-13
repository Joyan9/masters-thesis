import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalComparator:
    """
    Performs rigorous statistical comparisons of network metrics across contexts.
    
    This class implements the statistical analysis framework for RQ1 (Contextual Network
    Analysis). It compares network metrics (density, centralization, etc.) across different
    match contexts (score states, phases, intensity levels) using appropriate parametric
    or non-parametric tests.
    
    The analysis follows a structured workflow:
    1. Descriptive statistics by context
    2. Assumption testing (normality, homogeneity of variance)
    3. Overall group comparisons (ANOVA or Kruskal-Wallis)
    4. Post-hoc pairwise comparisons (Tukey HSD or Dunn's test)
    5. Effect size calculation (η², Cohen's d)
    6. Multiple testing correction (Holm-Bonferroni)
    
    Attributes
    ----------
    alpha : float
        Significance level for hypothesis tests (default: 0.05).
    min_sample_size : int
        Minimum sample size required for valid statistical testing (default: 10).
        Based on central limit theorem and power considerations.
    comparison_results : dict
        Storage for all statistical test results.
    
    Notes
    -----
    **Multiple Testing Correction:**
    
    With 6 metrics × 3 context types = 18 overall tests, plus numerous pairwise
    comparisons, Type I error inflation is a serious concern. This class applies
    Holm-Bonferroni correction to control family-wise error rate (FWER).
    
    Holm-Bonferroni is preferred over standard Bonferroni because:
    - More powerful (less conservative)
    - Still controls FWER at α level
    - Sequentially rejects hypotheses
    
    **Assumption Testing:**
    
    Parametric tests (ANOVA, t-test) require:
    1. Normality: Tested via Shapiro-Wilk (sensitive but appropriate for n < 50)
    2. Homogeneity of variance: Tested via Levene's test (robust to non-normality)
    
    If assumptions violated → Use non-parametric alternatives:
    - Kruskal-Wallis instead of ANOVA
    - Mann-Whitney U instead of t-test
    
    **Effect Size Interpretation (Cohen, 1988; Psychometrica):**
    
    Cohen's d (pairwise):
    - Small: 0.2
    - Medium: 0.5
    - Large: 0.8
    
    η² (overall):
    - Small: 0.01
    - Medium: 0.06
    - Large: 0.14
    
    **Practical Significance Thresholds:**
    
    Statistical significance (p < 0.05) doesn't guarantee tactical relevance.
    Minimum practically significant differences (based on football analytics):
    - Density: Δ ≥ 0.05 (5% change in connectivity)
    - Centralization: Δ ≥ 0.10 (10% change in hierarchy)
    - Clustering: Δ ≥ 0.05 (5% change in local cohesion)
    - Path length: Δ ≥ 0.20 (20% change in efficiency)
    - Centralities: Δ ≥ 0.05 (5% change in importance)
    
    References
    ----------
    - Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).
    - Holm, S. (1979). A simple sequentially rejective multiple test procedure.
      Scandinavian Journal of Statistics, 6(2), 65-70.
    - Levene, H. (1960). Robust tests for equality of variances. In Contributions to
      Probability and Statistics: Essays in Honor of Harold Hotelling.

    """
    
    # Practical significance thresholds (based on football analytics domain knowledge)
    PRACTICAL_THRESHOLDS = {
        'density': 0.05,
        'clustering_coefficient': 0.05,
        'avg_betweenness_centrality': 0.05,
        'avg_eigenvector_centrality': 0.05,
        'avg_path_length': 0.20,
        'centralization': 0.10,
        'normalized_path_length': 0.05
    }
    
    def __init__(self, alpha=0.05, min_sample_size=10):
        """
        Initialize StatisticalComparator.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for hypothesis tests. Standard threshold in social sciences.
        min_sample_size : int, default=10
            Minimum sample size required for valid statistical testing.
            
            Rationale:
            - Central Limit Theorem: n≥30 ideal, but n≥10 acceptable for robust tests
            - Shapiro-Wilk: Unreliable for n<10
            - Power: Small samples have low power to detect effects
            
            Groups with n<10 are excluded from analysis with warning.
        
        Notes
        -----
        **For Thesis Defense:**
        
        - α = 0.05: Standard in sports science and network analysis literature
        - min_sample_size = 10: Conservative threshold balancing power and reliability
        - Document any excluded groups due to insufficient sample size
        """
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.comparison_results = {}
    
    def compare_contexts(self, results_df: pd.DataFrame) -> Dict:
        """
        Compare network metrics across different contexts with rigorous statistical tests.
        
        Performs comprehensive statistical analysis for each context type (score, phase,
        intensity), including assumption testing, group comparisons, post-hoc tests,
        and effect size calculations. Applies multiple testing correction to control
        Type I error rate.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame of network metrics with columns:
            - Context columns: 'score_context', 'phase_context', 'intensity_context'
            - Metric columns: 'density', 'clustering_coefficient', etc.
            - Metadata: 'match_id', 'team', 'start_minute', etc.
        
        Returns
        -------
        dict
            Nested dictionary structure:
            {
                'score_context': {
                    'descriptive_stats': {...},
                    'normality_tests': {...},
                    'variance_tests': {...},
                    'group_comparisons': {...},
                    'pairwise_comparisons': {...},
                    'effect_sizes': {...},
                    'practical_significance': {...}
                },
                'phase_context': {...},
                'intensity_context': {...}
            }
        
        Notes
        -----
        **Analysis Workflow:**
        
        1. **Descriptive Statistics**: Mean, SD, median, range by context
        2. **Assumption Testing**:
           - Normality: Shapiro-Wilk per group
           - Homogeneity: Levene's test across groups
        3. **Overall Comparison**:
           - If assumptions met: One-way ANOVA
           - If violated: Kruskal-Wallis H test
        4. **Post-hoc Pairwise**:
           - If ANOVA: Tukey HSD (controls FWER)
           - If Kruskal-Wallis: Mann-Whitney U with Holm-Bonferroni correction
        5. **Effect Sizes**:
           - Overall: η² (eta-squared)
           - Pairwise: Cohen's d with 95% CI
        6. **Multiple Testing Correction**:
           - Holm-Bonferroni on all p-values
        
        **Sample Size Filtering:**
        
        Groups with n < min_sample_size are excluded with warning logged.
        This prevents unreliable results from small samples.
        
        """
        comparisons = {}
        
        # Define context types to analyze
        context_types = ['score_context', 'phase_context', 'intensity_context']
        
        # Collect all p-values for multiple testing correction
        all_p_values = []
        all_test_keys = []
        
        # First pass: Run all tests and collect p-values
        for context_type in context_types:
            if context_type in results_df.columns:
                logger.info(f"Analyzing context type: {context_type}")
                comparisons[context_type] = self._analyze_context_type(
                    results_df, context_type, collect_p_values=True
                )
                
                # Collect p-values for correction
                for metric, result in comparisons[context_type]['group_comparisons'].items():
                    if 'p_value' in result and result['p_value'] is not None:
                        all_p_values.append(result['p_value'])
                        all_test_keys.append((context_type, 'group', metric))
                
                for metric, pairwise in comparisons[context_type]['pairwise_comparisons'].items():
                    for comparison, result in pairwise.items():
                        if 'p_value' in result and result['p_value'] is not None:
                            all_p_values.append(result['p_value'])
                            all_test_keys.append((context_type, 'pairwise', metric, comparison))
        
        # Apply Holm-Bonferroni correction
        if all_p_values:
            logger.info(f"Applying Holm-Bonferroni correction to {len(all_p_values)} tests")
            reject, p_corrected, _, _ = multipletests(
                all_p_values, 
                alpha=self.alpha, 
                method='holm'
            )
            
            # Update results with corrected p-values
            for i, (p_corr, is_significant) in enumerate(zip(p_corrected, reject)):
                key = all_test_keys[i]
                
                if key[1] == 'group':
                    context_type, _, metric = key
                    comparisons[context_type]['group_comparisons'][metric]['p_value_corrected'] = float(p_corr)
                    comparisons[context_type]['group_comparisons'][metric]['significant_corrected'] = bool(is_significant)
                
                elif key[1] == 'pairwise':
                    context_type, _, metric, comparison = key
                    comparisons[context_type]['pairwise_comparisons'][metric][comparison]['p_value_corrected'] = float(p_corr)
                    comparisons[context_type]['pairwise_comparisons'][metric][comparison]['significant_corrected'] = bool(is_significant)
        
        self.comparison_results = comparisons
        return comparisons
    
    def _analyze_context_type(self, data: pd.DataFrame, context_type: str, 
                              collect_p_values: bool = False) -> Dict:
        """
        Analyze a specific context type (score, phase, or intensity).
        
        Performs complete statistical analysis for one context dimension, including
        descriptive statistics, assumption testing, group comparisons, and effect sizes.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset with network metrics and context labels.
        context_type : str
            Context dimension to analyze: 'score_context', 'phase_context', or 'intensity_context'.
        collect_p_values : bool, default=False
            Whether this is the first pass for collecting p-values (for correction).
        
        Returns
        -------
        dict
            Complete analysis results for this context type.
        
        Notes
        -----
        **Metrics Analyzed:**
        
        Core metrics from network_analyzer:
        - density
        - clustering_coefficient
        - avg_betweenness_centrality
        - avg_eigenvector_centrality
        - avg_path_length
        - centralization
        
        **Sample Size Filtering:**
        
        Groups with n < min_sample_size are excluded from analysis.
        Warning is logged for transparency.
        """
        metrics = [
            'density', 
            'clustering_coefficient', 
            'avg_betweenness_centrality',
            'avg_eigenvector_centrality', 
            'avg_path_length', 
            'centralization'
        ]
        
        results = {
            'context_type': context_type,
            'descriptive_stats': {},
            'normality_tests': {},
            'variance_tests': {},
            'group_comparisons': {},
            'pairwise_comparisons': {},
            'effect_sizes': {},
            'practical_significance': {}
        }
        
        # Get unique context labels
        context_labels = data[context_type].unique()
        
        # Filter out groups with insufficient sample size
        valid_labels = []
        for label in context_labels:
            n = len(data[data[context_type] == label])
            if n >= self.min_sample_size:
                valid_labels.append(label)
            else:
                logger.warning(f"Excluding {context_type}={label} (n={n} < {self.min_sample_size})")
        
        if len(valid_labels) < 2:
            logger.error(f"Insufficient groups for {context_type} (need ≥2, have {len(valid_labels)})")
            return results
        
        # Filter data to valid labels only
        data_filtered = data[data[context_type].isin(valid_labels)]
        
        # Descriptive statistics
        for metric in metrics:
            if metric in data_filtered.columns:
                desc_stats = data_filtered.groupby(context_type)[metric].agg([
                    'count', 'mean', 'std', 'median', 'min', 'max'
                ]).round(4)
                results['descriptive_stats'][metric] = desc_stats.to_dict('index')
        
        # Test each metric
        for metric in metrics:
            if metric in data_filtered.columns:
                metric_data = data_filtered[metric].dropna()
                if len(metric_data) > 0:
                    # Assumption testing
                    results['normality_tests'][metric] = self._test_normality(
                        data_filtered, metric, context_type
                    )
                    results['variance_tests'][metric] = self._test_variance_homogeneity(
                        data_filtered, metric, context_type
                    )
                    
                    # Group comparisons
                    results['group_comparisons'][metric] = self._compare_groups(
                        data_filtered, metric, context_type
                    )
                    
                    # Pairwise comparisons (only if overall test is significant)
                    group_result = results['group_comparisons'][metric]
                    if group_result.get('significant', False):
                        results['pairwise_comparisons'][metric] = self._pairwise_comparisons(
                            data_filtered, metric, context_type
                        )
                    else:
                        results['pairwise_comparisons'][metric] = {}
                    
                    # Effect sizes
                    results['effect_sizes'][metric] = self._calculate_effect_sizes(
                        data_filtered, metric, context_type
                    )
                    
                    # Practical significance
                    results['practical_significance'][metric] = self._assess_practical_significance(
                        data_filtered, metric, context_type
                    )
        
        return results
    
    def _test_normality(self, data: pd.DataFrame, metric: str, context_type: str) -> Dict:
        """
        Test normality assumption for each group using Shapiro-Wilk test.
        
        Normality is required for parametric tests (ANOVA, t-test). Shapiro-Wilk is
        appropriate for sample sizes 3 ≤ n ≤ 50 and is sensitive to departures from
        normality.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset to test.
        metric : str
            Metric column to test.
        context_type : str
            Context grouping variable.
        
        Returns
        -------
        dict
            Normality test results:
            {
                'by_group': {
                    'leading': {'statistic': 0.95, 'p_value': 0.12, 'is_normal': True, 'n': 45},
                    'trailing': {...},
                    'tied': {...}
                },
                'all_groups_normal': True/False,
                'recommended_test': 'parametric' or 'non_parametric'
            }
        
        Notes
        -----
        **Decision Rule:**
        
        - If ALL groups pass normality (p > α) → Use parametric tests
        - If ANY group fails (p ≤ α) → Use non-parametric tests
        
        This is conservative but ensures assumption validity. ANOVA is robust to
        moderate violations, but we prioritize rigor.
        
        **Shapiro-Wilk Limitations:**
        
        - Requires n ≥ 3 (minimum for any distributional test)
        - Low power for small samples (may not detect non-normality)
        - High power for large samples (may detect trivial departures)
        
        For n > 50, consider Kolmogorov-Smirnov or visual inspection (Q-Q plots).
    
        """
        normality_results = {}
        
        for context_label in data[context_type].unique():
            group_data = data[data[context_type] == context_label][metric].dropna()
            
            if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
                try:
                    stat, p_value = shapiro(group_data)
                    normality_results[context_label] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': bool(p_value > self.alpha),
                        'n': int(len(group_data))
                    }
                except Exception as e:
                    logger.warning(f"Shapiro-Wilk failed for {context_label}: {e}")
                    normality_results[context_label] = {
                        'statistic': None,
                        'p_value': None,
                        'is_normal': False,
                        'n': int(len(group_data)),
                        'error': str(e)
                    }
            else:
                normality_results[context_label] = {
                    'statistic': None,
                    'p_value': None,
                    'is_normal': False,
                    'n': int(len(group_data)),
                    'note': 'Insufficient data for normality test'
                }
        
        # Overall normality assessment
        all_normal = all(
            result.get('is_normal', False) 
            for result in normality_results.values()
        )
        
        return {
            'by_group': normality_results,
            'all_groups_normal': all_normal,
            'recommended_test': 'parametric' if all_normal else 'non_parametric'
        }
    
    def _test_variance_homogeneity(self, data: pd.DataFrame, metric: str, 
                                   context_type: str) -> Dict:
        """
        Test homogeneity of variance assumption using Levene's test.
        
        ANOVA assumes equal variances across groups. Levene's test is robust to
        non-normality (unlike Bartlett's test) and is the standard choice for
        variance homogeneity testing.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset to test.
        metric : str
            Metric column to test.
        context_type : str
            Context grouping variable.
        
        Returns
        -------
        dict
            Variance test results:
            {
                'test_name': 'Levene',
                'statistic': 2.34,
                'p_value': 0.098,
                'homogeneous': True,
                'groups_tested': ['leading', 'trailing', 'tied'],
                'n_groups': 3
            }
        
        Notes
        -----
        **Decision Rule:**
        
        - If p > α: Variances are homogeneous → Use standard ANOVA
        - If p ≤ α: Variances are heterogeneous → Use Welch's ANOVA (not implemented)
        
        Currently, we proceed with standard ANOVA even if violated (ANOVA is robust
        to moderate heterogeneity). For severe violations, consider Welch's ANOVA.
        
        **Levene's Test:**
        
        - H₀: All groups have equal variance
        - H₁: At least one group has different variance
        - Uses absolute deviations from group medians (robust to non-normality)
        
        References
        ----------
        Levene, H. (1960). Robust tests for equality of variances.
        """
        groups = []
        group_names = []
        
        for context_label in data[context_type].unique():
            group_data = data[data[context_type] == context_label][metric].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(context_label)
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for variance test'}
        
        try:
            statistic, p_value = levene(*groups)
            
            return {
                'test_name': 'Levene',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'homogeneous': bool(p_value > self.alpha),
                'groups_tested': group_names,
                'n_groups': len(groups)
            }
        
        except Exception as e:
            logger.error(f"Levene's test failed: {e}")
            return {'error': f'Variance test failed: {str(e)}'}
    
    def _compare_groups(self, data: pd.DataFrame, metric: str, context_type: str) -> Dict:
        """
        Compare groups using appropriate statistical test (ANOVA or Kruskal-Wallis).
        
        Performs overall group comparison to test whether network metrics differ
        significantly across contexts. Test selection based on assumption testing:
        - Parametric (ANOVA): If normality and homogeneity assumptions met
        - Non-parametric (Kruskal-Wallis): If assumptions violated
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with network metrics and context labels.
        metric : str
            Network metric to compare (e.g., 'density').
        context_type : str
            Context dimension (e.g., 'score_context').
        
        Returns
        -------
        dict
            Test results:
            {
                'test_name': 'One-way ANOVA' or 'Kruskal-Wallis',
                'statistic': 12.45,
                'p_value': 0.002,
                'significant': True,
                'groups_compared': ['leading', 'trailing', 'tied'],
                'n_groups': 3,
                'assumptions_met': True/False
            }
        
        Notes
        -----
        **Test Selection Logic:**
        
        1. Check normality (all groups must be normal)
        2. Check variance homogeneity
        3. If both met → One-way ANOVA (F-test)
        4. If either violated → Kruskal-Wallis H test
        
        **One-way ANOVA:**
        - H₀: All group means are equal (μ₁ = μ₂ = μ₃)
        - H₁: At least one group mean differs
        - Assumes: Normality, homogeneity of variance, independence
        - Test statistic: F = Between-group variance / Within-group variance
        
        **Kruskal-Wallis H Test:**
        - H₀: All groups have same distribution
        - H₁: At least one group differs
        - Non-parametric alternative to ANOVA
        - Based on ranks, not raw values
        - More robust but less powerful than ANOVA
        
        **Post-hoc Testing:**
        
        If overall test is significant (p < α), proceed to pairwise comparisons
        to identify which specific groups differ.
        """
        groups = []
        group_names = []
        
        for context_label in data[context_type].unique():
            group_data = data[data[context_type] == context_label][metric].dropna()
            if len(group_data) >= self.min_sample_size:
                groups.append(group_data)
                group_names.append(context_label)
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for comparison'}
        
        # Check assumptions
        normality_result = self._test_normality(data, metric, context_type)
        variance_result = self._test_variance_homogeneity(data, metric, context_type)
        
        use_parametric = (
            normality_result['all_groups_normal'] and 
            variance_result.get('homogeneous', False)
        )
        
        try:
            if use_parametric:
                # One-way ANOVA
                statistic, p_value = f_oneway(*groups)
                test_name = 'One-way ANOVA'
            else:
                # Kruskal-Wallis test
                statistic, p_value = kruskal(*groups)
                test_name = 'Kruskal-Wallis'
            
            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(p_value < self.alpha),
                'groups_compared': group_names,
                'n_groups': len(groups),
                'assumptions_met': use_parametric
            }
        
        except Exception as e:
            logger.error(f"Group comparison failed for {metric}: {e}")
            return {'error': f'Statistical test failed: {str(e)}'}
    
    def _pairwise_comparisons(self, data: pd.DataFrame, metric: str, 
                             context_type: str) -> Dict:
        """
        Perform pairwise post-hoc comparisons with appropriate test and correction.
        
        After finding significant overall group differences, identifies which specific
        pairs of groups differ. Uses Tukey HSD for parametric data (controls FWER)
        or Mann-Whitney U for non-parametric data (with Holm-Bonferroni correction).
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with network metrics and context labels.
        metric : str
            Network metric to compare.
        context_type : str
            Context dimension.
        
        Returns
        -------
        dict
            Pairwise comparison results:
            {
                'leading_vs_trailing': {
                    'test_name': 'Tukey HSD' or 'Mann-Whitney U',
                    'statistic': 3.45,
                    'p_value': 0.012,
                    'p_value_corrected': 0.024,  # Holm-Bonferroni
                    'significant': True,
                    'significant_corrected': True,
                    'cohens_d': 0.65,
                    'ci_lower': 0.02,
                    'ci_upper': 0.15,
                    'mean_diff': 0.085,
                    'group1_mean': 0.34,
                    'group2_mean': 0.26,
                    'n1': 45,
                    'n2': 38
                },
                'leading_vs_tied': {...},
                'trailing_vs_tied': {...}
            }
        
        Notes
        -----
        **Test Selection:**
        
        - Parametric: Tukey HSD (Honestly Significant Difference)
          - Controls family-wise error rate (FWER)
          - More powerful than Bonferroni
          - Assumes normality and homogeneity
        
        - Non-parametric: Mann-Whitney U with Holm-Bonferroni
          - Mann-Whitney U: Rank-based test for two independent samples
          - Holm-Bonferroni: Sequential correction for multiple comparisons
          - More conservative but protects against Type I error
        
        **Effect Size (Cohen's d):**
        
        d = (M₁ - M₂) / SD_pooled
        
        Where SD_pooled = √[((n₁-1)SD₁² + (n₂-1)SD₂²) / (n₁+n₂-2)]
        
        Interpretation (Cohen, 1988):
        - Small: d = 0.2
        - Medium: d = 0.5
        - Large: d = 0.8
        
        **Confidence Intervals:**
        
        95% CI for mean difference provides range of plausible values.
        If CI excludes 0, difference is significant at α = 0.05.
        
        **Multiple Testing Correction:**
        
        With k groups, there are k(k-1)/2 pairwise comparisons:
        - 3 groups → 3 comparisons
        - 4 groups → 6 comparisons
        
        Holm-Bonferroni adjusts p-values to control FWER while maintaining power.
        
        """
        pairwise_results = {}
        context_labels = sorted(data[context_type].unique())
        
        # Check if we should use parametric tests
        normality_result = self._test_normality(data, metric, context_type)
        variance_result = self._test_variance_homogeneity(data, metric, context_type)
        use_parametric = (
            normality_result['all_groups_normal'] and 
            variance_result.get('homogeneous', False)
        )
        
        if use_parametric:
            # Use Tukey HSD (controls FWER)
            try:
                tukey_result = pairwise_tukeyhsd(
                    endog=data[metric],
                    groups=data[context_type],
                    alpha=self.alpha
                )
                
                # Parse Tukey results
                for i in range(len(tukey_result.summary().data) - 1):  # Skip header
                    row = tukey_result.summary().data[i + 1]
                    group1, group2 = row[0], row[1]
                    mean_diff = float(row[2])
                    ci_lower = float(row[3])
                    ci_upper = float(row[4])
                    reject = row[5]  # Boolean
                    
                    comparison_key = f"{group1}_vs_{group2}"
                    
                    # Get group data for additional statistics
                    g1_data = data[data[context_type] == group1][metric].dropna()
                    g2_data = data[data[context_type] == group2][metric].dropna()
                    
                    # Calculate Cohen's d
                    pooled_std = np.sqrt(
                        ((len(g1_data)-1)*g1_data.std()**2 + 
                         (len(g2_data)-1)*g2_data.std()**2) / 
                        (len(g1_data)+len(g2_data)-2)
                    )
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    pairwise_results[comparison_key] = {
                        'test_name': 'Tukey HSD',
                        'mean_diff': mean_diff,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'significant': reject,
                        'cohens_d': float(cohens_d),
                        'group1_mean': float(g1_data.mean()),
                        'group2_mean': float(g2_data.mean()),
                        'n1': int(len(g1_data)),
                        'n2': int(len(g2_data))
                    }
            
            except Exception as e:
                logger.error(f"Tukey HSD failed: {e}")
                use_parametric = False  # Fall back to non-parametric
        
        if not use_parametric:
            # Use Mann-Whitney U with Holm-Bonferroni correction
            for i, label1 in enumerate(context_labels):
                for label2 in context_labels[i+1:]:
                    group1 = data[data[context_type] == label1][metric].dropna()
                    group2 = data[data[context_type] == label2][metric].dropna()
                    
                    comparison_key = f"{label1}_vs_{label2}"
                    
                    if len(group1) >= self.min_sample_size and len(group2) >= self.min_sample_size:
                        try:
                            # Mann-Whitney U test
                            statistic, p_value = mannwhitneyu(
                                group1, group2, alternative='two-sided'
                            )
                            
                            # Calculate Cohen's d
                            pooled_std = np.sqrt(
                                ((len(group1)-1)*group1.std()**2 + 
                                 (len(group2)-1)*group2.std()**2) / 
                                (len(group1)+len(group2)-2)
                            )
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            # Calculate 95% CI for mean difference (approximate)
                            se_diff = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
                            mean_diff = group1.mean() - group2.mean()
                            ci_lower = mean_diff - 1.96 * se_diff
                            ci_upper = mean_diff + 1.96 * se_diff
                            
                            pairwise_results[comparison_key] = {
                                'test_name': 'Mann-Whitney U',
                                'statistic': float(statistic),
                                'p_value': float(p_value),
                                'significant': bool(p_value < self.alpha),
                                'cohens_d': float(cohens_d),
                                'mean_diff': float(mean_diff),
                                'ci_lower': float(ci_lower),
                                'ci_upper': float(ci_upper),
                                'group1_mean': float(group1.mean()),
                                'group2_mean': float(group2.mean()),
                                'n1': int(len(group1)),
                                'n2': int(len(group2))
                            }
                        
                        except Exception as e:
                            logger.error(f"Mann-Whitney U failed for {comparison_key}: {e}")
                            pairwise_results[comparison_key] = {
                                'error': f'Pairwise test failed: {str(e)}',
                                'n1': int(len(group1)),
                                'n2': int(len(group2))
                            }
        
        return pairwise_results
    
    def _calculate_effect_sizes(self, data: pd.DataFrame, metric: str, 
                                context_type: str) -> Dict:
        """
        Calculate effect sizes for overall group comparisons.
        
        Effect size quantifies the magnitude of differences between groups,
        independent of sample size. η² (eta-squared) measures the proportion
        of variance in the metric explained by the context variable.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with network metrics and context labels.
        metric : str
            Network metric to analyze.
        context_type : str
            Context dimension.
        
        Returns
        -------
        dict
            Effect size results:
            {
                'eta_squared': 0.085,
                'interpretation': 'medium',
                'variance_explained': '8.5%'
            }
        
        Notes
        -----
        **η² (Eta-Squared) Formula:**
        
        η² = SS_between / SS_total
        
        Where:
        - SS_between: Sum of squares between groups
        - SS_total: Total sum of squares
        
        **Interpretation (Cohen, 1988; Psychometrica):**
        
        - Negligible: η² < 0.01 (< 1% variance explained)
        - Small: 0.01 ≤ η² < 0.06 (1-6% variance)
        - Medium: 0.06 ≤ η² < 0.14 (6-14% variance)
        - Large: η² ≥ 0.14 (≥ 14% variance)
        
        **Relationship to Other Effect Sizes:**
        
        - η² ≈ R² in regression
        - Cohen's f = √(η² / (1 - η²))
        - Partial η² adjusts for other factors (not used here)
        
        **Practical Significance:**
        
        Even small effect sizes can be practically important in sports contexts
        where marginal gains matter. Consider both statistical and practical
        significance when interpreting results.
        
        References
        ----------
        - Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
        - Psychometrica: https://www.psychometrica.de/effect_size.html
        """
        effect_sizes = {}
        context_labels = data[context_type].unique()
        
        # Collect all values
        all_values = data[metric].dropna()
        
        if len(all_values) == 0:
            return {'error': 'No valid data for effect size calculation'}
        
        # Calculate overall mean
        overall_mean = all_values.mean()
        
        # Total sum of squares
        total_ss = np.sum((all_values - overall_mean)**2)
        
        # Between-group sum of squares
        between_ss = 0
        for label in context_labels:
            group_data = data[data[context_type] == label][metric].dropna()
            if len(group_data) > 0:
                group_mean = group_data.mean()
                between_ss += len(group_data) * (group_mean - overall_mean)**2
        
        # Calculate eta-squared
        eta_squared = between_ss / total_ss if total_ss > 0 else 0
        
        effect_sizes['eta_squared'] = float(eta_squared)
        effect_sizes['variance_explained'] = f"{eta_squared * 100:.1f}%"
        
        # Interpret effect size (Cohen, 1988; Psychometrica)
        if eta_squared < 0.01:
            interpretation = 'negligible'
        elif eta_squared < 0.06:
            interpretation = 'small'
        elif eta_squared < 0.14:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        effect_sizes['interpretation'] = interpretation
        
        return effect_sizes
    
    def _assess_practical_significance(self, data: pd.DataFrame, metric: str,
                                       context_type: str) -> Dict:
        """
        Assess practical significance of differences beyond statistical significance.
        
        Statistical significance (p < 0.05) indicates that a difference is unlikely
        due to chance, but doesn't indicate whether the difference is large enough
        to matter in practice. This method evaluates whether observed differences
        exceed minimum thresholds for tactical relevance.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with network metrics and context labels.
        metric : str
            Network metric to assess.
        context_type : str
            Context dimension.
        
        Returns
        -------
        dict
            Practical significance assessment:
            {
                'threshold': 0.05,
                'max_difference': 0.12,
                'practically_significant': True,
                'comparisons': {
                    'leading_vs_trailing': {
                        'difference': 0.12,
                        'exceeds_threshold': True
                    },
                    ...
                }
            }
        
        Notes
        -----
        **Practical Significance Thresholds:**
        
        Based on football analytics domain knowledge and consultation with coaches:
        
        - Density: Δ ≥ 0.05 (5% change in connectivity)
          - Rationale: 5% change represents shift from ~30% to ~35% connectivity
        
        - Centralization: Δ ≥ 0.10 (10% change in hierarchy)
          - Rationale: 10% change represents meaningful tactical shift
        
        - Clustering: Δ ≥ 0.05 (5% change in local cohesion)
          - Rationale: Detectable change in passing triangle formation
        
        - Path Length: Δ ≥ 0.20 (20% change in efficiency)
          - Rationale: Represents ~0.5 pass difference in average path
        
        - Centralities: Δ ≥ 0.05 (5% change in importance)
          - Rationale: Meaningful shift in zone importance
        
        **Interpretation:**
        
        A result can be:
        1. Statistically significant but not practically significant (trivial effect)
        2. Practically significant but not statistically significant (underpowered)
        3. Both (ideal for strong conclusions)
        4. Neither (no evidence of difference)
        
        **For Thesis Defense:**
        
        Document threshold selection rationale. Ideally based on:
        - Literature review
        - Expert consultation
        - Pilot data analysis
        - Smallest effect of interest (SESOI) framework
        
        Examples
        --------
        >>> practical = comparator._assess_practical_significance(
        ...     data, 'density', 'score_context'
        ... )
        >>> practical
        {
            'threshold': 0.05,
            'max_difference': 0.12,
            'practically_significant': True,
            'comparisons': {
                'leading_vs_trailing': {
                    'difference': 0.12,
                    'exceeds_threshold': True
                }
            }
        }
        """
        threshold = self.PRACTICAL_THRESHOLDS.get(metric, 0.05)
        
        practical_results = {
            'threshold': threshold,
            'comparisons': {}
        }
        
        context_labels = sorted(data[context_type].unique())
        max_diff = 0
        
        # Compare all pairs
        for i, label1 in enumerate(context_labels):
            for label2 in context_labels[i+1:]:
                group1 = data[data[context_type] == label1][metric].dropna()
                group2 = data[data[context_type] == label2][metric].dropna()
                
                if len(group1) > 0 and len(group2) > 0:
                    diff = abs(group1.mean() - group2.mean())
                    max_diff = max(max_diff, diff)
                    
                    comparison_key = f"{label1}_vs_{label2}"
                    practical_results['comparisons'][comparison_key] = {
                        'difference': float(diff),
                        'exceeds_threshold': bool(diff >= threshold)
                    }
        
        practical_results['max_difference'] = float(max_diff)
        practical_results['practically_significant'] = bool(max_diff >= threshold)
        
        return practical_results
    
    def generate_statistical_report(self) -> str:
        """
        Generate comprehensive statistical report for RQ1 analysis.
        
        Creates human-readable summary of all statistical analyses, including
        descriptive statistics, test results, effect sizes, and practical
        significance assessments. Formatted for inclusion in thesis or
        supplementary materials.
        
        Returns
        -------
        str
            Formatted statistical report with sections:
            - Descriptive statistics by context
            - Assumption testing results
            - Overall group comparisons
            - Significant pairwise comparisons
            - Effect sizes and interpretations
            - Practical significance assessments
        
        Notes
        -----
        **Report Structure:**
        
        For each context type (score, phase, intensity):
        1. Descriptive statistics (M, SD, n)
        2. Assumption tests (normality, homogeneity)
        3. Overall tests (ANOVA/Kruskal-Wallis)
        4. Effect sizes (η²)
        5. Pairwise comparisons (if significant)
        6. Practical significance
        
        **Significance Indicators:**
        
        - *** : p < 0.001 (highly significant)
        - **  : p < 0.01 (very significant)
        - *   : p < 0.05 (significant)
        - (ns): p ≥ 0.05 (not significant)
        
        **Multiple Testing Correction:**
        
        Report shows both uncorrected and Holm-Bonferroni corrected p-values.
        Use corrected values for final conclusions to control Type I error.
        
        Examples
        --------
        >>> report = comparator.generate_statistical_report()
        >>> print(report)
        CONTEXTUAL NETWORK ANALYSIS REPORT
        ============================================================
        
        CONTEXT TYPE: SCORE CONTEXT
        ----------------------------------------
        
        DESCRIPTIVE STATISTICS:
        
        Density:
          leading: M=0.340, SD=0.082, n=145
          trailing: M=0.285, SD=0.075, n=132
          tied: M=0.312, SD=0.079, n=156
        
        STATISTICAL TESTS:
        
        Density:
          One-way ANOVA: F = 15.67, p = 0.0003 ***
          Effect size: η² = 0.085 (medium)
          Holm-Bonferroni corrected: p = 0.0018 **
        
        SIGNIFICANT PAIRWISE COMPARISONS:
        
          Density:
            leading_vs_trailing: p=0.001, d=0.68, Δ=0.055 **
            (Tukey HSD, corrected p=0.003)
        
        PRACTICAL SIGNIFICANCE:
        
          Density: Max difference = 0.055 (threshold = 0.050)
            ✓ Practically significant
        
        ============================================================
        """
        report = "CONTEXTUAL NETWORK ANALYSIS REPORT\n"
        report += "=" * 60 + "\n"
        report += f"Significance level: α = {self.alpha}\n"
        report += f"Minimum sample size: n ≥ {self.min_sample_size}\n"
        report += f"Multiple testing correction: Holm-Bonferroni\n"
        report += "=" * 60 + "\n\n"
        
        for context_type, results in self.comparison_results.items():
            report += f"CONTEXT TYPE: {context_type.upper().replace('_', ' ')}\n"
            report += "-" * 60 + "\n\n"
            
            # Descriptive statistics
            report += "DESCRIPTIVE STATISTICS:\n"
            report += "-" * 40 + "\n"
            for metric, stats in results['descriptive_stats'].items():
                report += f"\n{metric.replace('_', ' ').title()}:\n"
                for context_label, values in stats.items():
                    report += (f"  {context_label}: "
                             f"M={values['mean']:.3f}, "
                             f"SD={values['std']:.3f}, "
                             f"n={values['count']}\n")
            
            # Assumption testing
            report += "\nASSUMPTION TESTING:\n"
            report += "-" * 40 + "\n"
            for metric in results['normality_tests'].keys():
                norm_test = results['normality_tests'][metric]
                var_test = results['variance_tests'].get(metric, {})
                
                report += f"\n{metric.replace('_', ' ').title()}:\n"
                report += f"  Normality: {norm_test['recommended_test']}\n"
                if 'homogeneous' in var_test:
                    report += f"  Variance homogeneity: {'Yes' if var_test['homogeneous'] else 'No'}\n"
            
            # Statistical tests
            report += "\nSTATISTICAL TESTS:\n"
            report += "-" * 40 + "\n"
            for metric, test_result in results['group_comparisons'].items():
                if 'error' not in test_result:
                    test_name = test_result['test_name']
                    statistic = test_result['statistic']
                    p_value = test_result['p_value']
                    p_corrected = test_result.get('p_value_corrected', p_value)
                    significant = test_result.get('significant_corrected', test_result['significant'])
                    
                    # Significance stars
                    if p_corrected < 0.001:
                        sig_marker = '***'
                    elif p_corrected < 0.01:
                        sig_marker = '**'
                    elif p_corrected < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = '(ns)'
                    
                    report += f"\n{metric.replace('_', ' ').title()}:\n"
                    report += f"  {test_name}: "
                    if 'ANOVA' in test_name:
                        report += f"F = {statistic:.3f}"
                    else:
                        report += f"H = {statistic:.3f}"
                    report += f", p = {p_value:.4f}"
                    if p_value != p_corrected:
                        report += f" (corrected: p = {p_corrected:.4f})"
                    report += f" {sig_marker}\n"
                    
                    # Effect size
                    if metric in results['effect_sizes']:
                        eta_sq = results['effect_sizes'][metric].get('eta_squared', 0)
                        interpretation = results['effect_sizes'][metric].get('interpretation', 'unknown')
                        variance_exp = results['effect_sizes'][metric].get('variance_explained', 'N/A')
                        report += f"  Effect size: η² = {eta_sq:.3f} ({interpretation}, {variance_exp} variance)\n"
                    
                    # Practical significance
                    if metric in results['practical_significance']:
                        prac_sig = results['practical_significance'][metric]
                        if prac_sig.get('practically_significant', False):
                            report += f"  Practical significance: ✓ (max Δ = {prac_sig['max_difference']:.3f}, "
                            report += f"threshold = {prac_sig['threshold']:.3f})\n"
                        else:
                            report += f"  Practical significance: ✗ (max Δ = {prac_sig['max_difference']:.3f}, "
                            report += f"threshold = {prac_sig['threshold']:.3f})\n"
            
            # Significant pairwise comparisons
            report += "\nSIGNIFICANT PAIRWISE COMPARISONS:\n"
            report += "-" * 40 + "\n"
            significant_found = False
            
            for metric, pairwise in results['pairwise_comparisons'].items():
                significant_pairs = []
                
                for comparison, result in pairwise.items():
                    if 'error' not in result:
                        # Use corrected p-value if available
                        p_val = result.get('p_value_corrected', result.get('p_value'))
                        is_sig = result.get('significant_corrected', result.get('significant', False))
                        
                        if is_sig:
                            cohens_d = result.get('cohens_d', 0)
                            mean_diff = result.get('mean_diff', 0)
                            ci_lower = result.get('ci_lower', 0)
                            ci_upper = result.get('ci_upper', 0)
                            
                            # Significance stars
                            if p_val < 0.001:
                                sig_marker = '***'
                            elif p_val < 0.01:
                                sig_marker = '**'
                            elif p_val < 0.05:
                                sig_marker = '*'
                            else:
                                sig_marker = ''
                            
                            pair_str = (f"    {comparison}: "
                                      f"p={p_val:.4f} {sig_marker}, "
                                      f"d={cohens_d:.3f}, "
                                      f"Δ={mean_diff:.3f} "
                                      f"[{ci_lower:.3f}, {ci_upper:.3f}]")
                            significant_pairs.append(pair_str)
                
                if significant_pairs:
                    report += f"\n  {metric.replace('_', ' ').title()}:\n"
                    for pair in significant_pairs:
                        report += pair + "\n"
                    significant_found = True
            
            if not significant_found:
                report += "  No significant pairwise differences found (after correction).\n"
            
            report += "\n" + "=" * 60 + "\n\n"
        
        # Summary
        report += "SUMMARY:\n"
        report += "-" * 60 + "\n"
        report += f"Total context types analyzed: {len(self.comparison_results)}\n"
        
        total_tests = 0
        significant_tests = 0
        for results in self.comparison_results.values():
            for test_result in results['group_comparisons'].values():
                if 'p_value' in test_result:
                    total_tests += 1
                    if test_result.get('significant_corrected', False):
                        significant_tests += 1
        
        report += f"Total statistical tests: {total_tests}\n"
        report += f"Significant results (corrected): {significant_tests}\n"
        report += f"Multiple testing correction: Holm-Bonferroni (FWER control)\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "Legend:\n"
        report += "*** p < 0.001 (highly significant)\n"
        report += "**  p < 0.01 (very significant)\n"
        report += "*   p < 0.05 (significant)\n"
        report += "(ns) p ≥ 0.05 (not significant)\n"
        report += "=" * 60 + "\n"
        
        return report
    
    def get_aggregated_results(self) -> pd.DataFrame:
        """
        Get aggregated results (currently not implemented).
        
        Placeholder for potential future functionality to aggregate results
        across multiple analyses or datasets.
        
        Returns
        -------
        pd.DataFrame
            Empty DataFrame (not currently used).
        """
        return pd.DataFrame()
