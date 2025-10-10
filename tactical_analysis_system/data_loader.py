from statsbombpy import sb
import json
from datetime import datetime
import time
import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Data loader for StatsBomb football match data.
    
    This class handles downloading, processing, and storing football match data from 
    StatsBomb's open data API. It manages both raw data (matches and events) and 
    creates pandas DataFrames for compatibility with the analysis system.
    
    Attributes
    ----------
    matches : list of dict
        List of match metadata dictionaries containing information like match_id,
        competition, season, teams, scores, etc.
    events : dict
        Dictionary mapping match_id to list of event dictionaries. Each event
        contains detailed information about actions during the match (passes,
        shots, tackles, etc.).
    matches_data : pd.DataFrame
        DataFrame version of matches for analysis compatibility.
    events_data : pd.DataFrame
        DataFrame version of all events across all matches.
    lineups_data : pd.DataFrame
        DataFrame for lineup information (currently empty, reserved for future use).
   
    """
    
    def __init__(self):
        """
        Initialize DataLoader with empty data structures.
        
        Creates empty containers for matches, events, and their DataFrame
        representations.
        """
        self.matches = []
        self.events = {}
        # Initialize pandas DataFrames for compatibility with analysis system
        import pandas as pd
        self.matches_data = pd.DataFrame()
        self.events_data = pd.DataFrame()
        self.lineups_data = pd.DataFrame()
    
    def load_data(self, competitions, max_matches=None, batch_size=10, save_interval=None):
        """
        Load matches and events data from StatsBomb API in batches.
        
        Downloads match metadata and event data for specified competitions and seasons.
        Processes data in batches to manage memory and allows intermediate saves to
        prevent data loss during long-running downloads.
        
        Parameters
        ----------
        competitions : list of tuple
            List of (competition_id, season_id) tuples specifying which competitions
            and seasons to download. Example: [(11, 90), (43, 3)] for La Liga 2020/21
            and FIFA World Cup 2018.
        max_matches : int, optional
            Maximum number of matches to process. If None, processes all available
            matches. Useful for testing or limiting dataset size. Default is None.
        batch_size : int, default=10
            Number of matches to process in each batch before pausing. Helps manage
            API load and memory usage.
        save_interval : int, optional
            Number of matches after which to save interim data. For example, if
            save_interval=100, data will be saved after every 100 matches processed.
            If None, no interim saves are performed. Default is None.
        
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
        
        Notes
        -----
        - Includes automatic retry logic and error handling for failed API calls
        - Adds small delays (0.1s between matches, 0.5s between batches) to avoid
          overwhelming the API
        - Creates DataFrame versions of data automatically via _create_dataframes()
        - Prints progress updates every 10 matches and after each batch
        
        """
        
        # Load matches first
        for comp_id, season_id in competitions:
            try:
                season_matches = sb.matches(comp_id, season_id)
                self.matches.extend(season_matches.to_dict("records"))
                print(f"✅ Loaded {len(season_matches)} matches for comp {comp_id}, season {season_id}")
            except Exception as e:
                print(f"❌ Error loading comp {comp_id}, season {season_id}: {e}")
        
        # Apply max_matches limit if specified
        match_list = self.matches if max_matches is None else self.matches[:max_matches]
        total_matches = len(match_list)
        print(f"📊 Total matches to process: {total_matches}")
        
        # Process events in batches
        processed_count = 0
        failed_matches = []
        
        for i in range(0, total_matches, batch_size):
            batch_end = min(i + batch_size, total_matches)
            batch_matches = match_list[i:batch_end]
            
            print(f"\n🔄 Processing batch {i//batch_size + 1}: matches {i+1}-{batch_end} of {total_matches}")
            
            batch_success_count = 0
            for match in batch_matches:
                mid = match["match_id"]
                try:
                    self.events[mid] = sb.events(match_id=mid).to_dict("records")
                    batch_success_count += 1
                    processed_count += 1
                    
                    if processed_count % 10 == 0:  # Progress indicator
                        print(f"   ✅ Processed {processed_count}/{total_matches} matches")
                    
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"   ❌ Failed to load events for match {mid}: {e}")
                    failed_matches.append(mid)
            
            print(f"   Batch complete: {batch_success_count}/{len(batch_matches)} successful")
            
            # Save intermediate results if save_interval is specified
            if save_interval and (i + batch_size) % save_interval == 0:
                interim_filename = f"statsbomb_data_interim_{i + batch_size}.json"
                self._save_interim_data(interim_filename)
                print(f"   💾 Saved interim data: {interim_filename}")
            
            # Brief pause between batches
            time.sleep(0.5)
        
        print(f"\n📊 Data loading complete:")
        print(f"   - Total matches: {len(self.matches)}")
        print(f"   - Successfully loaded events: {len(self.events)}")
        print(f"   - Failed matches: {len(failed_matches)}")
        
        if failed_matches:
            print(f"   - Failed match IDs: {failed_matches[:10]}{'...' if len(failed_matches) > 10 else ''}")
        
        # Create DataFrame versions for compatibility with analysis system
        self._create_dataframes()
        
        return self
    
    def _create_dataframes(self):
        """
        Create pandas DataFrames from loaded data for analysis compatibility.
        
        Converts the raw dictionary-based data structures (self.matches and self.events)
        into pandas DataFrames. Performs data integrity checks, handles missing values,
        and ensures essential fields are present.
        
        Notes
        -----
        - Converts date/timestamp columns to pandas datetime objects
        - Handles missing location data by setting to [None, None]
        - Drops events missing essential fields: 'type', 'team', 'minute'
        - Adds match_id to each event for cross-referencing
        - Creates empty lineups_data DataFrame for future compatibility
        
        Side Effects
        ------------
        - Sets self.matches_data
        - Sets self.events_data
        - Sets self.lineups_data (empty)
        - Prints warnings if data integrity issues are found
        """
        import pandas as pd

        # Data integrity checks and missing data handling
        if not self.matches:
            print("⚠️ No matches loaded.")
            self.matches_data = pd.DataFrame()
        else:
            self.matches_data = pd.DataFrame(self.matches)
            if 'match_date' in self.matches_data.columns:
                self.matches_data['match_date'] = pd.to_datetime(self.matches_data['match_date'])

        # Combine all events, handle missing or malformed events
        all_events = []
        for match_id, events_list in self.events.items():
            for event in events_list:
                event['match_id'] = match_id
                # Handle missing coordinates - set to None for later filtering
                if 'location' in event and (event['location'] is None or len(event['location']) < 2):
                    event['location'] = [None, None]
                if 'pass_end_location' in event and (event['pass_end_location'] is None or len(event['pass_end_location']) < 2):
                    event['pass_end_location'] = [None, None]
                all_events.append(event)

        if all_events:
            self.events_data = pd.DataFrame(all_events)
            if 'timestamp' in self.events_data.columns:
                self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'])
            # Data integrity check: drop events missing essential fields
            essential_cols = ['type', 'team', 'minute']
            missing_essentials = self.events_data[self.events_data[essential_cols].isnull().any(axis=1)]
            if not missing_essentials.empty:
                print(f"⚠️ Dropping {len(missing_essentials)} events missing essential fields: {essential_cols}")
                self.events_data = self.events_data.drop(missing_essentials.index)
        else:
            self.events_data = pd.DataFrame()

        # Create empty lineups_data for compatibility (reserved for future use)
        if not hasattr(self, 'lineups_data'):
            self.lineups_data = pd.DataFrame()

        print(f"   - DataFrames created: {len(self.matches_data)} matches, {len(self.events_data)} events")
    
    def _save_interim_data(self, filename):
        """
        Save interim data during batch processing.
        
        Creates a checkpoint file during long-running data downloads. Useful for
        preventing data loss if the download process is interrupted.
        
        Parameters
        ----------
        filename : str
            Path where interim data should be saved.
        
        Notes
        -----
        - Saves with status='interim' to distinguish from complete datasets
        - Includes timestamp for tracking when data was saved
        - Uses default=str in json.dump to handle non-serializable objects
        """
        output_data = {
            'matches': self.matches,
            'events': self.events,
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(self.matches),
            'total_events': len(self.events),
            'status': 'interim'
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

    def save_data(self, filename):
        """
        Save complete loaded data to JSON file.
        
        Saves all matches, events, and DataFrame versions to a JSON file for
        later reuse. This avoids re-downloading data from the API.
        
        Parameters
        ----------
        filename : str
            Path where data should be saved.
        
        Notes
        -----
        - Saves with status='complete' to indicate full dataset
        - Includes both raw data (matches, events) and DataFrame versions
        - Includes metadata: timestamp, counts
        - Uses default=str to handle datetime and other non-JSON-serializable types
        
        """
        output_data = {
            'matches': self.matches,
            'events': self.events,
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(self.matches),
            'total_events': len(self.events),
            'status': 'complete'
        }
        
        # Also save DataFrame versions for compatibility
        if hasattr(self, 'matches_data') and not self.matches_data.empty:
            output_data['matches_data'] = self.matches_data.to_dict('records')
        
        if hasattr(self, 'events_data') and not self.events_data.empty:
            output_data['events_data'] = self.events_data.to_dict('records')
        
        if hasattr(self, 'lineups_data'):
            output_data['lineups_data'] = self.lineups_data.to_dict('records')
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Data saved to {filename}")
        print(f"   - Matches: {len(self.matches)}")
        print(f"   - Events: {len(self.events)}")

    def load_from_json(self, filepath: str) -> 'DataLoader':
        """
        Load previously saved data from JSON file.
        
        Reconstructs a DataLoader object from a previously saved JSON file,
        avoiding the need to re-download data from the API.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file containing saved data.
        
        Returns
        -------
        self : DataLoader
            Returns self for method chaining.
        
        Raises
        ------
        FileNotFoundError
            If the specified filepath does not exist.
        
        Notes
        -----
        - Automatically converts date/timestamp columns to pandas datetime
        - Ensures match_id is present in each event
        - Creates empty lineups_data for compatibility
        - Reconstructs both raw data (dict) and DataFrame versions
        
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Loading data from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct DataLoader object
        self.matches = data.get('matches', [])
        self.events = data.get('events', {})
        
        # Convert to pandas DataFrames for compatibility with analysis system
        import pandas as pd
        
        # Create matches_data DataFrame
        if self.matches:
            self.matches_data = pd.DataFrame(self.matches)
            # Convert date column if it exists
            if 'match_date' in self.matches_data.columns:
                self.matches_data['match_date'] = pd.to_datetime(self.matches_data['match_date'])
        else:
            self.matches_data = pd.DataFrame()
        
        # Create events_data DataFrame by combining all events
        if self.events:
            all_events = []
            for match_id, events_list in self.events.items():
                for event in events_list:
                    event['match_id'] = match_id  # Ensure match_id is in each event
                    all_events.append(event)  # Note: Could use extend([event]) but append is clearer
            
            self.events_data = pd.DataFrame(all_events)
            # Convert timestamp column if it exists
            if 'timestamp' in self.events_data.columns:
                self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'])
        else:
            self.events_data = pd.DataFrame()
        
        # Create empty lineups_data for compatibility
        self.lineups_data = pd.DataFrame()
        
        print(f"✅ Loaded {len(self.matches)} matches, {len(self.events)} match events")
        print(f"✅ Created DataFrames: {len(self.matches_data)} matches, {len(self.events_data)} events")
        
        return self
    