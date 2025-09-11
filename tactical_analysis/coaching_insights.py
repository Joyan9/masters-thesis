"""
Coaching Insights Generator
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import json

class CoachingInsightsEngine:
    def __init__(self, network_analyzer, motif_analyzer):
        self.network_analyzer = network_analyzer
        self.motif_analyzer = motif_analyzer
        self.insights = {}
        self.rule_confidence = {}
        
        # Research-backed thresholds for decision rules
        self.thresholds = {
            'high_centrality': 0.15,      # BC > 0.15 indicates tactical key nodes (Clemente et al., 2015)
            'low_density': 0.3,           # Network density < 0.3 indicates disconnected play
            'high_clustering': 0.6,       # Clustering > 0.6 indicates over-reliance on local connections
            'motif_imbalance': 0.7,       # Motif ratio > 0.7 indicates tactical predictability
            'critical_time': 30,          # Last 30 minutes are tactically critical
            'pressure_threshold': 0.25    # Edge weight variance > 0.25 indicates pressure
        }
        
        # Coaching recommendation categories
        self.recommendations = {
            'formation': [
                'Switch to 3-5-2 for width',
                'Adopt 4-2-3-1 for control',
                'Use 4-4-2 for directness',
                'Try 3-4-3 for attacking overload'
            ],
            'playing_style': [
                'Switch to direct play',
                'Increase possession tempo',
                'Focus on wing utilization', 
                'Emphasize central penetration',
                'Adopt counter-attacking approach'
            ],
            'positioning': [
                'Push fullbacks higher',
                'Drop midfield deeper',
                'Narrow the formation',
                'Stretch the play wider',
                'Compact defensive shape'
            ],
            'tempo': [
                'Increase passing tempo',
                'Slow down build-up',
                'Quick transitions',
                'Patient possession',
                'High-intensity pressing'
            ]
        }
    
    def calculate_confidence_score(self, pattern_data, context_data):
        """
        Calculate confidence score for recommendations
        Multi-factor approach: Statistical significance (40%) + Sample size (30%) + Historical success (30%)
        """
        confidence_factors = {}
        
        # Factor 1: Statistical Significance (40% weight)
        p_value = pattern_data.get('p_value', 1.0)
        if p_value < 0.001:
            stat_score = 1.0
        elif p_value < 0.01:
            stat_score = 0.8
        elif p_value < 0.05:
            stat_score = 0.6
        else:
            stat_score = 0.2
        
        confidence_factors['statistical_significance'] = stat_score * 0.4
        
        # Factor 2: Sample Size Adequacy (30% weight)
        sample_size = pattern_data.get('sample_size', 0)
        if sample_size >= 20:
            sample_score = 1.0
        elif sample_size >= 10:
            sample_score = 0.7
        elif sample_size >= 5:
            sample_score = 0.4
        else:
            sample_score = 0.1
        
        confidence_factors['sample_adequacy'] = sample_score * 0.3
        
        # Factor 3: Historical Success Rate (30% weight)
        # Based on context similarity and typical success rates
        context_score = self.estimate_historical_success(context_data)
        confidence_factors['historical_success'] = context_score * 0.3
        
        # Total confidence score
        total_confidence = sum(confidence_factors.values())
        
        return {
            'total_confidence': min(1.0, total_confidence),
            'factors': confidence_factors,
            'confidence_level': self.categorize_confidence(total_confidence)
        }
    
    def estimate_historical_success(self, context_data):
        """Estimate historical success rate based on context similarity"""
        # Research-backed success rates for different contexts
        base_rates = {
            'Leading': 0.75,    # High success rate when ahead
            'Tied': 0.60,       # Moderate success rate
            'Trailing': 0.45,   # Lower success rate when behind
        }
        
        score_context = context_data.get('score_context', 'Tied')
        team_quality = context_data.get('team_quality', 'Middle 10')
        match_phase = context_data.get('match_phase', 'Middle')
        
        base_rate = base_rates.get(score_context, 0.60)
        
        # Adjust for team quality
        if team_quality == 'Top 5':
            base_rate += 0.1
        elif team_quality == 'Bottom 5':
            base_rate -= 0.1
        
        # Adjust for match phase
        if match_phase == 'Late':
            base_rate -= 0.05  # More difficult in late stages
        
        return max(0.1, min(0.9, base_rate))
    
    def categorize_confidence(self, confidence_score):
        """Categorize confidence score into levels"""
        if confidence_score >= 0.8:
            return 'Very High'
        elif confidence_score >= 0.6:
            return 'High'
        elif confidence_score >= 0.4:
            return 'Medium'
        elif confidence_score >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def generate_formation_insights(self, team_metrics, context):
        """Generate formation-related insights"""
        insights = []
        
        density = team_metrics.get('density', 0)
        clustering = team_metrics.get('average_clustering', 0)
        bc_dc_ratio = team_metrics.get('bc_dc_ratio_mean', 0)
        
        # Rule 1: Low density suggests formation spread
        if density < self.thresholds['low_density']:
            insights.append({
                'type': 'formation',
                'recommendation': 'Narrow the formation',
                'reasoning': f'Network density ({density:.3f}) indicates disconnected play',
                'urgency': 'High' if density < 0.2 else 'Medium',
                'pattern_data': {'density': density, 'threshold': self.thresholds['low_density']}
            })
        
        # Rule 2: High clustering suggests over-reliance on local connections
        if clustering > self.thresholds['high_clustering']:
            insights.append({
                'type': 'formation', 
                'recommendation': 'Stretch the play wider',
                'reasoning': f'High clustering ({clustering:.3f}) indicates over-reliance on local connections',
                'urgency': 'Medium',
                'pattern_data': {'clustering': clustering, 'threshold': self.thresholds['high_clustering']}
            })
        
        return insights
    
    def generate_playing_style_insights(self, team_metrics, context, motif_data=None):
        """Generate playing style insights"""
        insights = []
        
        bc_dc_ratio = team_metrics.get('bc_dc_ratio_mean', 0)
        edge_variance = team_metrics.get('edge_weight_variance', 0)
        score_context = context.get('score_context', 'Tied')
        time_remaining = 90 - context.get('time_window', [0, 90])[1]
        
        # Rule 3: High centrality + trailing + late game = direct play
        if (bc_dc_ratio > self.thresholds['high_centrality'] and 
            score_context == 'Trailing' and 
            time_remaining < self.thresholds['critical_time']):
            
            insights.append({
                'type': 'playing_style',
                'recommendation': 'Switch to direct play',
                'reasoning': f'High centrality ({bc_dc_ratio:.3f}) + trailing + {time_remaining} min remaining',
                'urgency': 'Very High',
                'pattern_data': {
                    'bc_dc_ratio': bc_dc_ratio,
                    'score_context': score_context,
                    'time_remaining': time_remaining
                }
            })
        
        # Rule 4: High edge variance indicates pressure
        if edge_variance > self.thresholds['pressure_threshold']:
            if score_context == 'Leading':
                recommendation = 'Patient possession'
            else:
                recommendation = 'Quick transitions'
            
            insights.append({
                'type': 'playing_style',
                'recommendation': recommendation,
                'reasoning': f'High edge variance ({edge_variance:.3f}) indicates pressure situation',
                'urgency': 'High',
                'pattern_data': {'edge_variance': edge_variance, 'threshold': self.thresholds['pressure_threshold']}
            })
        
        # Rule 5: Motif-based insights
        if motif_data:
            total_motifs = sum(motif_data.values())
            if total_motifs > 0:
                triangle_ratio = motif_data.get('triangle_cycle', 0) / total_motifs
                
                if triangle_ratio > self.thresholds['motif_imbalance']:
                    insights.append({
                        'type': 'playing_style',
                        'recommendation': 'Increase wing utilization',
                        'reasoning': f'Over-reliance on central circulation ({triangle_ratio:.3f})',
                        'urgency': 'Medium',
                        'pattern_data': {'triangle_ratio': triangle_ratio, 'total_motifs': total_motifs}
                    })
        
        return insights
    
    def generate_positioning_insights(self, team_metrics, context):
        """Generate player positioning insights"""
        insights = []
        
        density = team_metrics.get('density', 0)
        bc_dc_ratio = team_metrics.get('bc_dc_ratio_mean', 0)
        score_context = context.get('score_context', 'Tied')
        
        # Rule 6: Low density + leading = compact shape
        if density < self.thresholds['low_density'] and score_context == 'Leading':
            insights.append({
                'type': 'positioning',
                'recommendation': 'Compact defensive shape',
                'reasoning': f'Low density ({density:.3f}) while leading suggests need for compactness',
                'urgency': 'Medium',
                'pattern_data': {'density': density, 'score_context': score_context}
            })
        
        # Rule 7: High centrality + trailing = push players forward
        if bc_dc_ratio > self.thresholds['high_centrality'] and score_context == 'Trailing':
            insights.append({
                'type': 'positioning',
                'recommendation': 'Push fullbacks higher',
                'reasoning': f'High centrality ({bc_dc_ratio:.3f}) + trailing suggests need for width',
                'urgency': 'High',
                'pattern_data': {'bc_dc_ratio': bc_dc_ratio, 'score_context': score_context}
            })
        
        return insights
    
    def generate_tempo_insights(self, team_metrics, context):
        """Generate tempo-related insights"""
        insights = []
        
        edge_variance = team_metrics.get('edge_weight_variance', 0)
        density = team_metrics.get('density', 0)
        score_context = context.get('score_context', 'Tied')
        match_phase = context.get('match_phase', 'Middle')
        
        # Rule 8: Low variance + trailing = increase tempo
        if edge_variance < 0.1 and score_context == 'Trailing':
            insights.append({
                'type': 'tempo',
                'recommendation': 'Increase passing tempo',
                'reasoning': f'Low variance ({edge_variance:.3f}) + trailing suggests predictable play',
                'urgency': 'High',
                'pattern_data': {'edge_variance': edge_variance, 'score_context': score_context}
            })
        
        # Rule 9: High density + leading + late = slow tempo
        if (density > 0.5 and score_context == 'Leading' and match_phase == 'Late'):
            insights.append({
                'type': 'tempo',
                'recommendation': 'Slow down build-up',
                'reasoning': f'High density ({density:.3f}) + leading + late game suggests time management',
                'urgency': 'Medium',
                'pattern_data': {'density': density, 'score_context': score_context, 'match_phase': match_phase}
            })
        
        return insights
    
    def generate_match_insights(self, match_id):
        """Generate comprehensive insights for a match"""
        if match_id not in self.network_analyzer.network_metrics:
            print(f"No network metrics for match {match_id}")
            return None
        
        if match_id not in self.network_analyzer.context_classifier.context_classifications:
            print(f"No context classifications for match {match_id}")
            return None
        
        match_contexts = self.network_analyzer.context_classifier.context_classifications[match_id]
        match_info = match_contexts['match_info']
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        match_insights = {
            'match_id': match_id,
            'match_info': match_info,
            'team_insights': {}
        }
        
        for team in [home_team, away_team]:
            team_insights = {
                'formation': [],
                'playing_style': [],
                'positioning': [],
                'tempo': []
            }
            
            # Analyze each context window
            for phase_name, phase_data in match_contexts['contexts'].items():
                if phase_name == 'Full' or team not in phase_data:
                    continue
                
                context = phase_data[team]
                
                # Get network metrics for this context
                if (team in self.network_analyzer.network_metrics[match_id] and
                    phase_name in self.network_analyzer.network_metrics[match_id][team]):
                    
                    team_metrics = self.network_analyzer.network_metrics[match_id][team][phase_name]
                    
                    if team_metrics:
                        # Get motif data if available
                        motif_data = None
                        if (match_id in self.motif_analyzer.motif_patterns and
                            team in self.motif_analyzer.motif_patterns[match_id].get('context_analysis', {})):
                            
                            team_contexts = self.motif_analyzer.motif_patterns[match_id]['context_analysis'][team]
                            for context_key, context_motif_data in team_contexts.items():
                                if context_motif_data['context']['match_phase'] == context['match_phase']:
                                    motif_data = context_motif_data['motifs']
                                    break
                        
                        # Generate insights for each category
                        formation_insights = self.generate_formation_insights(team_metrics, context)
                        style_insights = self.generate_playing_style_insights(team_metrics, context, motif_data)
                        positioning_insights = self.generate_positioning_insights(team_metrics, context)
                        tempo_insights = self.generate_tempo_insights(team_metrics, context)
                        
                        # Add confidence scores and context info
                        for insight_list, category in [(formation_insights, 'formation'),
                                                     (style_insights, 'playing_style'),
                                                     (positioning_insights, 'positioning'),
                                                     (tempo_insights, 'tempo')]:
                            
                            for insight in insight_list:
                                confidence = self.calculate_confidence_score(
                                    insight['pattern_data'], context
                                )
                                insight['confidence'] = confidence
                                insight['context'] = {
                                    'phase': phase_name,
                                    'score_context': context['score_context'],
                                    'match_phase': context['match_phase'],
                                    'team_quality': context['team_quality']
                                }
                                
                                team_insights[category].append(insight)
            
            match_insights['team_insights'][team] = team_insights
        
        self.insights[match_id] = match_insights
        return match_insights
    
    def prioritize_insights(self, team_insights):
        """Prioritize insights by urgency and confidence"""
        all_insights = []
        
        for category, insights in team_insights.items():
            for insight in insights:
                insight['category'] = category
                all_insights.append(insight)
        
        # Sort by urgency and confidence
        urgency_order = {'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1}
        
        sorted_insights = sorted(all_insights, 
                               key=lambda x: (urgency_order.get(x['urgency'], 0),
                                            x['confidence']['total_confidence']),
                               reverse=True)
        
        return sorted_insights
    
    def generate_coaching_report(self, match_id):
        """Generate a comprehensive coaching report"""
        if match_id not in self.insights:
            self.generate_match_insights(match_id)
        
        if match_id not in self.insights:
            return None
        
        match_insights = self.insights[match_id]
        match_info = match_insights['match_info']
        
        print(f"\n{'='*60}")
        print(f"COACHING INSIGHTS REPORT - MATCH {match_id}")
        print(f"{'='*60}")
        print(f"Teams: {match_info['home_team']} vs {match_info['away_team']}")
        print(f"Final Score: {match_info['final_score']}")
        
        for team, team_insights in match_insights['team_insights'].items():
            print(f"\n{'-'*40}")
            print(f"TEAM: {team}")
            print(f"{'-'*40}")
            
            prioritized = self.prioritize_insights(team_insights)
            
            if not prioritized:
                print("No specific insights generated for this team.")
                continue
            
            print(f"\nTOP PRIORITY RECOMMENDATIONS:")
            
            for i, insight in enumerate(prioritized[:5], 1):  # Top 5 recommendations
                confidence = insight['confidence']
                print(f"\n{i}. {insight['recommendation'].upper()}")
                print(f"   Category: {insight['category'].title()}")
                print(f"   Urgency: {insight['urgency']}")
                print(f"   Confidence: {confidence['confidence_level']} ({confidence['total_confidence']:.2f})")
                print(f"   Reasoning: {insight['reasoning']}")
                print(f"   Context: {insight['context']['match_phase']} phase, {insight['context']['score_context']}")
        
        return match_insights
    
    def process_multiple_matches(self, match_ids=None):
        """Process coaching insights for multiple matches"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        print(f"\n=== GENERATING COACHING INSIGHTS FOR {len(match_ids)} MATCHES ===")
        
        for i, match_id in enumerate(match_ids, 1):
            print(f"Processing insights {i}/{len(match_ids)}: {match_id}")
            self.generate_match_insights(match_id)
        
        print("✅ Coaching insights generation complete!")
        return self.insights
    
    def save_coaching_insights(self, filename='coaching_insights_days8_9.json'):
        """Save coaching insights"""
        with open(filename, 'w') as f:
            json.dump(self.insights, f, indent=2, default=str)
        
        print(f"✅ Coaching insights saved to {filename}")
