import json
import pandas as pd
import os

class Reader():
    def __init__(self, filepath='', laurel=1):
        self.filepath = filepath
        self.first_index = 0
        self.step = 1000
        self.second_index = 1000


    def get_attrs(self):
        print("=======================")
        print(f"Current filepath : {self.filepath}")
        print(f"First index : {self.first_index}")
        print(f"Second index : {self.second_index}")
        print(f"Step : {self.step}")

    def reset_attrs(self, first_index=0, second_index=1000, step=1000):
        self.first_index = first_index
        self.second_index = second_index
        self.step = step
        print("Resetting values")
        self.get_attrs()

    def check_file(self, filepath):
        # Check if file exists
        # Print debug information
        # Raise error if file not found
        if not os.path.exists(self.filepath):
            print(f"\n[-] Error, file {self.filepath} does not exist.")
            current_directory = os.getcwd()
            print(f"\n[*] Current Directory: {current_directory}")
            print("\n[*] Files in the current directory:")
            for file_name in os.listdir(current_directory):
                if os.path.isfile(os.path.join(current_directory, file_name)):
                    print(file_name)

            raise FileNotFoundError


    def count_lines(self, filepath):
        self.filepath = filepath
        self.check_file(self.filepath)

        i = 0
        with open(self.filepath, 'r') as file:
            for line in file:
                i+=1
        return i


    def read_file(self, filepath, return_list=False):
        self.filepath = filepath
        self.check_file(self.filepath)
        data_list = []
        lines_limit = 1000 # limit reading lines, done for memory constraints
        i = 0

        with open(self.filepath, 'r') as file:
            for line in file:
                i+=1
                if i > lines_limit:
                    break
                # Find the index of the first '{'
                index = line.find('{')
                if index != -1:
                    json_data = line[index:]
                    data = json.loads(json_data)
                    data_list.append(data)

        if return_list:
            return data_list

        df = pd.DataFrame(data_list)
        return df

    
    def read_chunk(self, filepath, step=1000):

        self.filepath = filepath
        self.check_file(self.filepath)
        self.step = step
        self.second_index = self.first_index + self.step
        
        data_list = []
        i = 0
        with open(self.filepath, 'r') as file:
            for i,line in enumerate(file):
                if i >= self.first_index:
                    i+=1
                    if i > self.second_index:
                        break
                    # Find the index of the first '{'
                    index = line.find('{')
                    if index != -1:
                        json_data = line[index:]
                        data = json.loads(json_data)
                        data_list.append(data)
                else:
                    i += 1
        
        df = pd.DataFrame(data_list)
        
        self.first_index += self.step
        self.second_index += self.step
        
        return df


    def read(self, filepath):
        self.filepath = filepath
        self.check_file(self.filepath)
        
        n_lines = self.count_lines(self.filepath)
        lines_limit = 1000
        
        if n_lines < lines_limit:
            df = self.read_file(filepath)
        else:
            df = self.read_chunk(file_path)
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