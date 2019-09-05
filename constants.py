METRICS = ['recency', 'frequency', 'monetary']
clustering_parameters = {'split_ratio': 0.8, 'seed': 1234, 'k': 4,
                         'hyper_params': {'standardize': [True, False], 'init': ['Random', 'Furthest', 'PlusPlus']},
                         'search_criteria': {'strategy': "Cartesian"}


                         }