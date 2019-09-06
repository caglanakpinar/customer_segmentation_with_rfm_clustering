import pandas as pd

def gather_data(parameters):
    path_rfm = parameters['file_path_rfm'] if parameters['file_path_rfm'] else 'rfm_values.csv'
    path_prev_rfm = parameters['file_path_prev_day'] if parameters['file_path_prev_day'] else 'prev_day_segments.csv'
    return  pd.read_csv(path_rfm), pd.read_csv(path_prev_rfm)

def write_to_csv(data):
    data.to_csv('rfm_scores.csv')
