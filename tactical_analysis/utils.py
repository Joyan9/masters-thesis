import numpy as np
import warnings
warnings.filterwarnings("ignore")

def map_coordinates_to_zone(x, y, pitch_length=120, pitch_width=80):
    """Map pitch coordinates to 7x7 grid zones (49 zones total)"""
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return None
    
    # Normalize coordinates to 0-1 range
    x_norm = max(0, min(1, x / pitch_length))
    y_norm = max(0, min(1, y / pitch_width))
    
    # Map to 7x7 grid
    zone_x = min(6, int(x_norm * 7))
    zone_y = min(6, int(y_norm * 7))
    
    # Return zone ID (0-48)
    zone_id = zone_y * 7 + zone_x
    return zone_id

def create_rolling_windows(match_duration=90, window_size=10, step_size=5):
    """Create rolling time windows for dynamic analysis"""
    windows = []
    start = 0
    while start + window_size <= match_duration:
        windows.append((start, start + window_size))
        start += step_size
    return windows

# Add other shared utility functions here
