METRICS = ['recency', 'frequency', 'monetary']
METRIC_VALUES = {'recency': ['recency', 'rec', 'r'],
                 'frequency': ['frequency', 'freq', 'f'],
                 'monetary': ['monetary', 'mon', 'm']
                 }
Z_VALUES = [-1.96, 1.96] # 95% OF CONFIDENCE
clustering_parameters = {'split_ratio': 0.8, 'seed': 1234, 'k': 4,
                         'hyper_params': {'standardize': [True, False], 'init': ['Random', 'Furthest', 'PlusPlus']},
                         'search_criteria': {'strategy': "Cartesian"}

                         }
accepted_minimum_prob = 0.05