import json
import pandas as pd


#file_path = 'sample_logs_cron_extended.txt'

def parse_file_to_df(file_path):
    
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Find the index of the first '{'
            index = line.find('{')
            if index != -1:
                json_data = line[index:]
                data = json.loads(json_data)
                data_list.append(data)

    df = pd.DataFrame(data_list)
    #print(df)
    #print(df.describe())
    return df