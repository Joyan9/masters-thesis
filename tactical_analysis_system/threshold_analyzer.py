import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

class ThresholdAnalyzer:
    """Extract performance thresholds from RQ1 results for rule-based recommendations"""
    
    def __init__(self, rq1_results: Dict = None):
        self.rq1_results = rq1_results
        self.thresholds = {}
        self.performance_profiles = {}
        
    def extract_performance_thresholds(self, network_data: pd.DataFrame, 
                                     outcome_column: str = 'team_performance') -> Dict:
        """
        Extract performance-based thresholds from network metrics
        
        Args:
            network_data: DataFrame with network metrics and performance indicators
            outcome_column: Column indicating team performance (win/loss/draw or rating)
        """
        print("Extracting performance thresholds from RQ1 results...")
        
        # Define network metrics
        metrics = [
            'density', 'clustering_coefficient', 'avg_betweenness_centrality',
            'avg_eigenvector_centrality', 'avg_path_length', 'centralization'
        ]
        
        available_metrics = [m for m in metrics if m in network_data.columns]
        
        thresholds = {}
        
        for metric in available_metrics:
            metric_thresholds = self._calculate_metric_thresholds(
                network_data, metric, outcome_column
            )
            thresholds[metric] = metric_thresholds
            
        # Add context-specific thresholds
        context_thresholds = self._extract_context_thresholds(network_data)
        thresholds.update(context_thresholds)
        
        # Add intensity-based thresholds (from RQ1 findings)
        intensity_thresholds = self._extract_intensity_thresholds(network_data)
        thresholds.update(intensity_thresholds)
        
        self.thresholds = thresholds
        return thresholds
    
    def _calculate_metric_thresholds(self, data: pd.DataFrame, 
                                   metric: str, outcome_column: str) -> Dict:
        """Calculate percentile-based thresholds for a specific metric"""
        
        metric_data = data[metric].dropna()
        
        # Calculate percentile thresholds
        percentiles = {
            'excellent': np.percentile(metric_data, 90),    # Top 10%
            'good': np.percentile(metric_data, 75),         # Top 25%
            'average': np.percentile(metric_data, 50),      # Median
            'poor': np.percentile(metric_data, 25),         # Bottom 25%
            'critical': np.percentile(metric_data, 10)      # Bottom 10%
        }
        
        # Add statistical thresholds based on distribution
        mean_val = metric_data.mean()
        std_val = metric_data.std()
        
        statistical_thresholds = {
            'high_performance': mean_val + std_val,         # 1 SD above mean
            'low_performance': mean_val - std_val,          # 1 SD below mean
            'extreme_high': mean_val + 2 * std_val,        # 2 SD above mean
            'extreme_low': mean_val - 2 * std_val          # 2 SD below mean
        }
        
        # Combine thresholds
        thresholds = {
            'percentiles': percentiles,
            'statistical': statistical_thresholds,
            'mean': mean_val,
            'std': std_val,
            'range': {
                'min': metric_data.min(),
                'max': metric_data.max()
            }
        }
        
        return thresholds
    
    def _extract_context_thresholds(self, data: pd.DataFrame) -> Dict:
        """Extract context-specific thresholds from RQ1 findings"""
        
        context_thresholds = {}
        
        # Intensity context thresholds (major finding from RQ1)
        if 'intensity_context' in data.columns:
            intensity_thresholds = {}
            
            for intensity in data['intensity_context'].unique():
                if pd.notna(intensity):
                    intensity_data = data[data['intensity_context'] == intensity]
                    
                    intensity_thresholds[intensity] = {
                        'density_mean': intensity_data['density'].mean() if 'density' in data.columns else None,
                        'clustering_mean': intensity_data['clustering_coefficient'].mean() if 'clustering_coefficient' in data.columns else None,
                        'sample_size': len(intensity_data)
                    }
            
            context_thresholds['intensity_thresholds'] = intensity_thresholds
        
        # Phase context thresholds
        if 'phase_context' in data.columns:
            phase_thresholds = {}
            
            for phase in data['phase_context'].unique():
                if pd.notna(phase):
                    phase_data = data[data['phase_context'] == phase]
                    
                    phase_thresholds[phase] = {
                        'density_mean': phase_data['density'].mean() if 'density' in data.columns else None,
                        'path_length_mean': phase_data['avg_path_length'].mean() if 'avg_path_length' in data.columns else None,
                        'sample_size': len(phase_data)
                    }
            
            context_thresholds['phase_thresholds'] = phase_thresholds
        
        return context_thresholds
    
    def _extract_intensity_thresholds(self, data: pd.DataFrame) -> Dict:
        """Extract intensity-specific thresholds based on RQ1 major finding"""
        
        if 'intensity_context' not in data.columns:
            return {}
        
        # Based on RQ1: Intensity has large effect (η² = 0.388)
        intensity_profiles = {}
        
        for intensity in ['low', 'medium', 'high']:
            if intensity in data['intensity_context'].values:
                intensity_data = data[data['intensity_context'] == intensity]
                
                if len(intensity_data) > 0:
                    profile = {
                        'density_target': intensity_data['density'].mean() if 'density' in data.columns else None,
                        'clustering_target': intensity_data['clustering_coefficient'].mean() if 'clustering_coefficient' in data.columns else None,
                        'centralization_target': intensity_data['centralization'].mean() if 'centralization' in data.columns else None,
                        'sample_size': len(intensity_data),
                        'recommended_for': self._get_intensity_recommendations(intensity)
                    }
                    
                    intensity_profiles[intensity] = profile
        
        return {'intensity_profiles': intensity_profiles}
    
    def _get_intensity_recommendations(self, intensity: str) -> List[str]:
        """Get tactical recommendations for each intensity level"""
        
        recommendations = {
            'low': [
                'Maintain possession',
                'Build up slowly',
                'Control tempo',
                'Focus on ball retention'
            ],
            'medium': [
                'Balanced approach',
                'Mix short and long passes',
                'Moderate pressing',
                'Flexible positioning'
            ],
            'high': [
                'Quick transitions',
                'Direct play',
                'High pressing',
                'Fast ball circulation'
            ]
        }
        
        return recommendations.get(intensity, [])
    
    def save_thresholds(self, filepath: str = "results/performance_thresholds.json"):
        """Save extracted thresholds to file"""
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        thresholds_serializable = convert_numpy(self.thresholds)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(thresholds_serializable, f, indent=2)
        
        print(f"Thresholds saved to {filepath}")
    
    def load_thresholds(self, filepath: str = "results/performance_thresholds.json"):
        """Load thresholds from file"""
        
        try:
            with open(filepath, 'r') as f:
                self.thresholds = json.load(f)
            print(f"Thresholds loaded from {filepath}")
        except FileNotFoundError:
            print(f"Threshold file not found: {filepath}")
            self.thresholds = {}
    
    def get_threshold_summary(self) -> Dict:
        """Get summary of extracted thresholds"""
        
        if not self.thresholds:
            return {"error": "No thresholds available. Run extract_performance_thresholds first."}
        
        summary = {
            'total_metrics': len([k for k in self.thresholds.keys() 
                                if k not in ['intensity_profiles', 'intensity_thresholds', 'phase_thresholds']]),
            'context_profiles': len(self.thresholds.get('intensity_profiles', {})),
            'key_findings': []
        }
        
        # Add key findings from thresholds
        if 'density' in self.thresholds:
            density_thresholds = self.thresholds['density']['percentiles']
            summary['key_findings'].append(
                f"Density excellence threshold: {density_thresholds['excellent']:.3f}"
            )
        
        if 'intensity_profiles' in self.thresholds:
            intensity_profiles = self.thresholds['intensity_profiles']
            if 'high' in intensity_profiles and 'low' in intensity_profiles:
                high_density = intensity_profiles['high'].get('density_target', 0)
                low_density = intensity_profiles['low'].get('density_target', 0)
                if high_density and low_density:
                    improvement = ((high_density - low_density) / low_density) * 100
                    summary['key_findings'].append(
                        f"High intensity increases density by {improvement:.1f}%"
                    )
        
        return summary
