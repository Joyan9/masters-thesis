from tactical_analysis_system.data_loader import DataLoader

competitions = [(2, 44)]
dl = DataLoader().load_data(competitions, max_matches=None)
dl.save_data("sb_open_2018_19.json")
