import json
import pandas as pd


#file_path = 'sample_logs_cron_extended.txt'

def parse_file_to_df(file_path):
    
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            index = line.find('{')
            if index != -1:
                json_data = line[index:]
                data = json.loads(json_data)
                data_list.append(data)

    df = pd.DataFrame(data_list)
    return df

def parse_file_2(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data_string = line
            data_string = data_string.strip('"')
            data_string = data_string.replace('\"message\": "\\"', '"message": "')

            # Try parsing the corrected data as JSON
            try:
                valid_json = json.loads(data_string)
                print(valid_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")


def parse_line(line):
    index = line.find('{')
    if index != -1:
        json_data = line[index:]
        data = json.loads(json_data)
        #print(data)

    return data