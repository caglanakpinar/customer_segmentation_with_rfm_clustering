import pandas as pd
import numpy as np
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.grid.grid_search import H2OGridSearch
from scipy.stats import norm

import constants

def KMeans_ClusteringH2O(data, metric, parameters):
    try:
        h2o.init()
        rfm_data = h2o.H2OFrame(data)
        train, valid = rfm_data.split_frame(ratios=[constants.clustering_parameters['split_ratio']],
                                            seed=constants.clustering_parameters['seed'])
        rfm_kmeans = H2OKMeansEstimator(k=constants.clustering_parameters['k'],
                                        seed=constants.clustering_parameters['seed'],
                                        max_iterations=int(len(data) / 2))
        rfm_kmeans.train(x=metric, training_frame=train, validation_frame=valid)
        grid = H2OGridSearch(model=rfm_kmeans, hyper_params=constants.clustering_parameters['hyper_params'],
                             search_criteria=constants.clustering_parameters['search_criteria'])
        # train using the grid
        grid.train(x=metric, training_frame=train, validation_frame=valid)

        # sort the grid models by total within cluster sum-of-square error.
        sorted_grid = grid.get_grid(sort_by='tot_withinss', decreasing=False)
        prediction = sorted_grid[0].predict(rfm_data)
        data = rfm_data.concat(prediction, axis=1)[[metric, 'predict']].as_data_frame(use_pandas=True)
        data = data.rename(columns={'predict': metric + '_segment'})
        data[metric + '_segment'] = data[metric + '_segment'].apply(lambda x: x + 1)
        if parameters['is_h2o_cluster_shut_down']:
            h2o.shutdown(prompt=False)
    except:
        if parameters['is_h2o_cluster_shut_down']:
            h2o.shutdown(prompt=False)
    return data

def current_day_r_f_m_clustering(data, parameters):
    for metric in constants.METRICS:
        clustered_data = KMeans_ClusteringH2O(data[[metric]], metric, parameters)
        clustered_data = clustered_data.drop(metric, axis=1)
        data = pd.concat([data, clustered_data], axis=1)
    return data

def get_segment_changes(data, metric, metric_values):
    # Let`s assign each value which are on Critical Point from which segment to which segment
    df_list = []
    metric_str = metric_values[1]
    for row in data.to_dict('results'):
        d = row
        d['changing_segment_' + metric_str] = '-'
        if row['critical_lower_' + metric_str] != '-':
            if row[metric] > row['critical_lower_' + metric_str]:
                if row[metric] < row['critical_upper_' + metric_str]:
                    d['changing_segment_' + metric_str] = str(row[metric+'_segment']) + \
                                                          '_' + str(row[metric + '_segment'] + 1)
        df_list.append(d)

    return pd.DataFrame(df_list)

def get_segment_descriptive_stats(data, segment, metric, metric_values):
    segment_values = list(data[data[metric+'_segment'] == segment][metric])
    # confidence intervals for critical points
    right_tail_critical_point = np.mean(segment_values) + constants.Z_VALUES[1] * np.std(segment_values)
    left_tail_critical_point = np.mean(segment_values) + constants.Z_VALUES[0] * np.std(segment_values)
    data['critical_lower_' + metric_values[1]] = left_tail_critical_point
    data['critical_upper_' + metric_values[1]] = right_tail_critical_point
    data[metric_values[1] + '_mean'] = np.mean(segment_values)
    data[metric_values[1] + '_std'] = np.std(segment_values)
    data = get_segment_changes(data, metric, metric_values)
    return data

def compute_segment_change(data, prev_data):
    segment_change_df = pd.DataFrame()
    for metric in constants.METRICS:
        for s in range(1, constants.clustering_parameters['k']):
            data = get_segment_descriptive_stats(data, s, metric, constants.METRIC_VALUES[metric])
            segment_change = get_transicion_matrix_with_probabilities(s, s+1, data, prev_data,
                                                                      constants.METRIC_VALUES[metric]
                                                                      )
            segment_change_df = segment_change if len(segment_change_df) == 0 else pd.concat([segment_change,
                                                                                                segment_change_df])
    return segment_change_df

# in order to calculate the segment change of probability
def standart_normal_dist_probability_calculation(value, mean, std):
    Z_values = (value - mean) / std
    prob = (norm.cdf(Z_values))
    return Z_values, prob

def get_transicion_matrix_with_probabilities(segment_1, segment_2, df, prev_df, values):
    df = df.query(values[0] + "_segment in (@segment_1, @segment_2)")
    changing_segment = str(segment_1) + '_' + str(segment_2)

    if len(df[df['changing_segment_' + values[1]] == changing_segment]) != 0:
        df = pd.merge(df, prev_df.rename(columns={'recency_segment': 'recency_segment_before',
                                                  'monetary_segment': 'monetary_segment_before',
                                                  'frequency_segment': 'frequency_segment_before'}),
                      on='client_id', how='left')
        asd = df[df[values[0] + '_segment_before'] == segment_1]
        query_str_1 = values[0] + "_segment == @segment_1 and " + values[0] + "_segment_before == @segment_1"
        query_str_2 = values[0] + "_segment == @segment_2 and " + values[0] + "_segment_before == @segment_2"
        if len(df[df[values[0] + '_segment_before'] == segment_1]) != 0:
            prob_segment_1_1 = len(df.query(query_str_1)) / len(df[df[values[0] + '_segment_before'] == segment_1])
        else:
            prob_segment_1_1 = 0.05
        prob_segment_1_2 = 1 - prob_segment_1_1
        if len(df[df[values[0] + '_segment_before'] == segment_2]) != 0:
            prob_segment_2_2 = len(df.query(query_str_2)) / len(df[df[values[0] + '_segment_before'] == segment_2])
        else:
            prob_segment_2_2 = constants.accepted_minimum_prob
        prob_segment_2_1 = 1 - prob_segment_2_2

        t_m_df = df.query("changing_segment_" + values[1] + " == @changing_segment")
        mean, std = np.mean(t_m_df[values[0]]), np.std(t_m_df[values[0]])

        # probability calculation of segment 1 to 2 change. It is calculated by Critical Interval 1 to 2 Distribution.
        t_m_df['probability_critic_2'] = t_m_df.apply(lambda row:
                                                      standart_normal_dist_probability_calculation(row[values[0]],
                                                                                                   mean,
                                                                                                   std
                                                                                                   )[1],
                                                      axis=1)
        # probability calculation of segment 2 to 1 change. It is calculated by Critical Interval 2 to 1 Distribution.
        t_m_df['probability_critic_1'] = t_m_df['probability_critic_2'].apply(lambda x: 1 - x)

        t_m_df[values[2] + '_segment_from'], t_m_df[values[2] + '_segment_to'] = segment_1, segment_2
        t_m_df[values[2] + '_segment_degrease_prob'] = (t_m_df['probability_critic_1'] * prob_segment_1_2) + \
                                                       (t_m_df['probability_critic_2'] * prob_segment_2_2)
        return t_m_df[
            ['client_id', values[2] + '_segment_degrease_prob', values[2] + '_segment_from', values[2] + '_segment_to']]
    else:
        return pd.DataFrame()