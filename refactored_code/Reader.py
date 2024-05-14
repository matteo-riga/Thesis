import json
import pandas as pd
import os

class Reader():
    def __init__(self, filepath, laurel = 0):
        self.filepath = filepath

        # Check if file exists
        # Print debug information
        # Raise error if file not found
        if not os.path.exists(self.filepath) and not laurel:
            print(f"\n[-] Error, file {self.filepath} does not exist.")
            current_directory = os.getcwd()
            print(f"\n[*] Current Directory: {current_directory}")
            print("\n[*] Files in the current directory:")
            for file_name in os.listdir(current_directory):
                if os.path.isfile(os.path.join(current_directory, file_name)):
                    print(file_name)

            raise FileNotFoundError

    # THIS FUNCTION DOES NOT WORK
    def read_line(self, line):
        index = line.find('{')
        if index != -1:
            json_data = line[index:]
            data = json.loads(json_data)
        
        line_df = pd.DataFrame(data)
        return line_df

    def read_file(self):

        data_list = []

        with open(self.filepath, 'r') as file:
            for line in file:
                # Find the index of the first '{'
                index = line.find('{')
                if index != -1:
                    json_data = line[index:]
                    data = json.loads(json_data)
                    data_list.append(data)

        df = pd.DataFrame(data_list)
        return df
    

    def parse_line(self, line):

        index = line.find('{')
        data = -1

        if index != -1:
            json_data = line[index:]
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return -1

        return data