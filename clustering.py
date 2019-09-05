import pandas as pd
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.grid.grid_search import H2OGridSearch

import constants

def KMeans_ClusteringH2O(data, metric):
    try:
        h2o.init()
        rfm_data = h2o.H2OFrame(data)
        train, valid = rfm_data.split_frame(ratios=[constants.clustering_parameters['split_ratio']],
                                            seed=constants.clustering_parameters['seed'])
        rfm_kmeans = H2OKMeansEstimator(k=constants.clustering_parameters['seed'],
                                        seed=constants.clustering_parameters['seed'],
                                        max_iterations=int(len(data) / 2))
        rfm_kmeans.train(x='recency', training_frame=train, validation_frame=valid)
        grid = H2OGridSearch(model=rfm_kmeans, hyper_params=constants.clustering_parameters['hyper_params'],
                             search_criteria=constants.clustering_parameters['search_criteria'])
        # train using the grid
        grid.train(x=metric, training_frame=train, validation_frame=valid)

        # sort the grid models by total within cluster sum-of-square error.
        sorted_grid = grid.get_grid(sort_by='tot_withinss', decreasing=False)
        prediction = sorted_grid[0].predict(rfm_data)
        rfm_data = rfm_data.concat(prediction, axis=1)[[metric, metric + '_segment'
                                                        ]].as_data_frame(use_pandas=True).rename(
            columns={'predict': 'cluster'})
        rfm_data[metric + '_segment'] = rfm_data[metric + '_segment'].apply(lambda x: x + 1)
        h2o.shutdown(prompt=True)
    except:
        h2o.shutdown(prompt=True)

    return rfm_data[[metric + '_segment']]

def current_day_r_f_m_clustering(data):
    for metric in constants.METRICS:
        clustered_data = KMeans_ClusteringH2O(data[metric], metric)
        data = pd.concat([data, clustered_data], axis=1)


def get_transicion_matrix_with_probabilities(segment_1, segment_2, df, prev_df, values):
    df = df.query(values[0] + "_segment in (@segment_1, @segment_2)")
    changing_segment = str(segment_1) + '_' + str(segment_2)

    if len(df[df['changing_segment_' + values[1]] == changing_segment]) != 0:
        df = pd.merge(df, prev_df, on='client_id', how='left')
        query_str_1 = values[0] + "_segment == @segment_1 and " + values[0] + "_segment_before == @segment_1"
        query_str_2 = values[0] + "_segment == @segment_2 and " + values[0] + "_segment_before == @segment_2"
        prob_segment_1_1 = len(df.query(query_str_1)) / len(df[df[values[0] + '_segment_before'] == segment_1])
        prob_segment_1_2 = 1 - prob_segment_1_1
        prob_segment_2_2 = len(df.query(query_str_2)) / len(df[df[values[0] + '_segment_before'] == segment_2])
        prob_segment_2_1 = 1 - prob_segment_2_2

        t_m_df = df.query("changing_segment_" + values[1] + " == @changing_segment")
        mean, std = np.mean(t_m_df[values[0]]), np.std(t_m_df[values[0]])

        # probability calculation of segment 1 to 2 change. It is calculated by Critical Interval 1 to 2 Distribution.
        t_m_df['probability_critic_2'] = t_m_df.apply(lambda row:
                                                      standart_normal_dist_probability_calculation(row[values[0]], mean,
                                                                                                   std)[1], axis=1)
        # probability calculation of segment 2 to 1 change. It is calculated by Critical Interval 2 to 1 Distribution.
        t_m_df['probability_critic_1'] = t_m_df['probability_critic_2'].apply(lambda x: 1 - x)

        t_m_df[values[2] + '_segment_from'], t_m_df[values[2] + '_segment_to'] = segment_1, segment_2
        t_m_df[values[2] + '_segment_degrease_prob'] = (t_m_df['probability_critic_1'] * prob_segment_1_2) + \
                                                       (t_m_df['probability_critic_2'] * prob_segment_2_2)
        return t_m_df[
            ['client_id', values[2] + '_segment_degrease_prob', values[2] + '_segment_from', values[2] + '_segment_to']]
    else:
        return pd.DataFrame()