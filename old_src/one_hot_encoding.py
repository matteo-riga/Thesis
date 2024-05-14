import numpy as np
import pandas as pd
from datetime import datetime

import parser_1
import dictionaries
import preprocess


def one_hot_encode_after_preprocessing():
    # general log preprocessing analysis for each type of log
    log_types = ['cron', 'laurel', 'maillog', 'messages', 'secure', 'user']
    file_paths = ['sample_logs_' + logtype + '.txt' for logtype in log_types]

    #print(file_paths)

    # Severity normal level (established arbitrarily)
    normal_level = "info"
    normal_level_numerical = dictionaries.severity_dict[normal_level]

    df_list = []

    for file_path in file_paths:
        df = parser_1.parse_file_to_df(file_path)
        # Create severity and facility ID numbers for each dataframe
        df['severity_numbers'] = [dictionaries.severity_dict[elem] for elem in df['severity']]
        df['facility_numbers'] = [dictionaries.facility_dict[elem] for elem in df['facility']]
        # Create severity scores
        df['severity_scores'] = [np.exp(normal_level_numerical-elem) for elem in df['severity_numbers']]
        # Append time deltas
        timedeltas = []
        for i, date_str in enumerate(df['time']):
            if i == 0:
                timedeltas.append(0)
            else:
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S %z')
                date_prev = datetime.strptime(df['time'][i-1], '%Y-%m-%d %H:%M:%S %z')
                timedelta = date - date_prev
                timedeltas.append(timedelta.total_seconds())

        df['timedelta'] = timedeltas
        df_list.append(df)

    for i, df in enumerate(df_list):
        func = preprocess.get_function(log_types[i])
        df = func(df)

    # SANITY CHECK
    '''
    for i, df in enumerate(df_list):
        print('==============')
        print(log_types[i])
        print(df.columns)
    '''

    columns_to_drop = ['message', 'severity', 'facility', 'time' ,'severity_numbers']
    columns_to_encode =['host', 'ident', 'pid', 'facility_numbers', 'user']

    encoded_df_list = []

    for i,df in enumerate(df_list):
        #print('================')
        for col in columns_to_drop:
            try:
                #print(f'removed {col} from df {i}')
                df = df.drop(columns=col)
                #print(f'shape {df.shape}')
            except:
                pass
            
        for col in columns_to_encode:
            try:
                df = pd.get_dummies(df, columns=[col])
                #print(f'encoded {col} from df {i}')
            except:
                pass
            
            #print(f'shape {df.shape}')
            
        encoded_df_list.append(df)

    return encoded_df_list