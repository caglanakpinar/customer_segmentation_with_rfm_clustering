import data_access
import clustering

parameters = {'file_path_rfm': None, 'file_path_prev_day': None}



def main(parameters):
    rfm_df, prev_day_df = data_access.gather_data(parameters)
    = clustering.current_day_r_f_m_clustering(rfm_df)



if __name__ == '__main__':
    main(parameters)