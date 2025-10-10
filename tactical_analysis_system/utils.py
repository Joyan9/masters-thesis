import numpy as np
import warnings
warnings.filterwarnings("ignore")

def map_coordinates_to_zone(x, y, period=None, pitch_length=120, pitch_width=80):
    """
    Map pitch coordinates to a 7x7 grid zone system (49 zones total).
    
    This function converts continuous pitch coordinates into discrete zones for 
    spatial analysis. The pitch is divided into a 7x7 grid, creating 49 zones 
    numbered from 0 to 48. Coordinates are normalized to ensure the team always 
    attacks in the same direction (left to right) regardless of match period.
    
    Parameters
    ----------
    x : float or None
        X-coordinate on the pitch. Expected range: [0, pitch_length].
        None or NaN values will return None.
    y : float or None
        Y-coordinate on the pitch. Expected range: [0, pitch_width].
        None or NaN values will return None.
    period : int, optional
        Match period (1 for first half, 2 for second half).
        If period == 2, coordinates are inverted so the team always attacks 
        left → right. Default is None.
    pitch_length : float, default=120
        Length of the pitch in the same units as x-coordinate (typically meters).
        Standard football pitch length.
    pitch_width : float, default=80
        Width of the pitch in the same units as y-coordinate (typically meters).
        Standard football pitch width.
    
    Returns
    -------
    int or None
        Zone ID ranging from 0 to 48, where:
        - Zone 0 is bottom-left (defensive left corner)
        - Zone 6 is bottom-right (defensive right corner)
        - Zone 42 is top-left (attacking left corner)
        - Zone 48 is top-right (attacking right corner)
        Returns None if input coordinates are invalid (None or NaN).
    
    """
    # Handle invalid coordinates
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return None
   
    # Flip coordinates for second half to maintain consistent attack direction
    # This ensures team always attacks from left (x=0) to right (x=120)
    if period == 2:
        x = pitch_length - x
        y = pitch_width - y
   
    # Normalize coordinates to 0–1 range
    # max/min clamps ensure coordinates outside pitch boundaries are handled
    x_norm = max(0, min(1, x / pitch_length))
    y_norm = max(0, min(1, y / pitch_width))
   
    # Map normalized coordinates to 7x7 grid indices
    # int() truncates, so 0.99 → 6, but we clamp to ensure max is 6
    zone_x = min(6, int(x_norm * 7))  # Column index (0-6)
    zone_y = min(6, int(y_norm * 7))  # Row index (0-6)
   
    # Convert 2D grid position to 1D zone ID
    # Row-major ordering: zone_id = row * num_columns + column
    zone_id = zone_y * 7 + zone_x
    return zone_id


def create_sliding_windows(match_duration=90, window_size=10, step_size=5):
    """
    Create overlapping time windows for dynamic temporal analysis.
    
    Generates a list of time intervals using a sliding window approach, where
    each window has a fixed size and windows overlap based on the step size.
    This is commonly used for analyzing how network metrics evolve over time.
    
    Parameters
    ----------
    match_duration : int or float, default=90
        Total duration of the match in minutes. Standard match duration is 90 minutes.
    window_size : int or float, default=10
        Size of each time window in minutes. Each window will span this duration.
    step_size : int or float, default=5
        Step size between consecutive windows in minutes. Determines overlap:
        - If step_size < window_size: windows overlap
        - If step_size == window_size: windows are adjacent (no overlap)
        - If step_size > window_size: gaps between windows
    
    Returns
    -------
    list of tuple
        List of (start_time, end_time) tuples representing each window.
        Times are in the same units as input parameters (minutes).
    
    """
    windows = []
    start = 0
    
    # Continue creating windows until we can't fit another complete window
    while start + window_size <= match_duration:
        windows.append((start, start + window_size))
        start += step_size
    
    return windows


def get_context_label(minute, score_diff):
    """
    Determine contextual labels for a given match state.
    
    Assigns categorical labels based on temporal phase and score situation,
    enabling context-aware analysis of tactical behavior. This function is
    central to RQ1's investigation of how network characteristics differ
    across match contexts.
    
    Parameters
    ----------
    minute : int or float
        Current match minute (0-90+ for regular time).
    score_diff : int or float
        Goal difference from the team's perspective:
        - Positive: team is leading
        - Negative: team is trailing
        - Zero: match is tied
            
    Returns
    -------
    dict
        Dictionary containing context labels with keys:
        - 'score' : str
            Score context, one of {'leading', 'trailing', 'tied'}
        - 'phase' : str
            Match phase, one of {'early', 'middle', 'late'}
    
    """
    contexts = {}
    
    # Determine score context based on goal difference
    if score_diff > 0:
        contexts['score'] = 'leading'
    elif score_diff < 0:
        contexts['score'] = 'trailing'
    else:
        contexts['score'] = 'tied'
    
    # Determine temporal phase (divides match into thirds)
    if minute <= 30:
        contexts['phase'] = 'early'
    elif minute <= 60:
        contexts['phase'] = 'middle'
    else:
        contexts['phase'] = 'late'
    
    return contexts
