import numpy as np
import warnings
warnings.filterwarnings("ignore")

def map_coordinates_to_zone(x, y, period=None, pitch_length=120, pitch_width=80):
    """Map pitch coordinates to 7x7 grid zones (49 zones total).
   
    If period == 2, invert coordinates so that the team always attacks left → right.
    """
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return None
   
    # Flip coordinates for second half
    if period == 2:
        x = pitch_length - x
        y = pitch_width - y
   
    # Normalize coordinates to 0–1 range
    x_norm = max(0, min(1, x / pitch_length))
    y_norm = max(0, min(1, y / pitch_width))
   
    # Map to 7x7 grid
    zone_x = min(6, int(x_norm * 7))
    zone_y = min(6, int(y_norm * 7))
   
    # Return zone ID (0–48)
    zone_id = zone_y * 7 + zone_x
    return zone_id

def create_sliding_windows(match_duration=90, window_size=10, step_size=5):
    """Create sliding time windows for dynamic analysis"""
    windows = []
    start = 0
    while start + window_size <= match_duration:
        windows.append((start, start + window_size))
        start += step_size
    return windows

def get_context_label(minute, score_diff, match_contexts):
    """Get context labels for a given minute and score difference"""
    contexts = {}
    
    # Score context
    if score_diff > 1:
        contexts['score'] = 'leading'
    elif score_diff < -1:
        contexts['score'] = 'trailing'
    else:
        contexts['score'] = 'tied'
    
    # Phase context
    if minute <= 30:
        contexts['phase'] = 'early'
    elif minute <= 60:
        contexts['phase'] = 'middle'
    else:
        contexts['phase'] = 'late'
    
    return contexts