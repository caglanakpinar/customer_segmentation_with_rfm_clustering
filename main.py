import data_access
import clustering

parameters = {'file_path_rfm': None, 'file_path_prev_day': None}



def main(parameters):
    rfm_df, prev_day_df = data_access.gather_data(parameters)
    rfm_df = clustering.current_day_r_f_m_clustering(rfm_df)
    rfm_df = clustering.compute_segment_change(rfm_df, prev_day_df)
    data_access.write_to_csv(rfm_df)

if __name__ == '__main__':
    main(parameters)