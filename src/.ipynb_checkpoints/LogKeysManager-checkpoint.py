import json
import os

class LogKeysManager():
    def __init__(self, file_path, temp_log_key):
        self.file_path = file_path
        self.temporary_log_key = temp_log_key

    def get_log_key_dict(self, log_key_dict_file_path):
        '''
        Function that retrieves the file containing the log keys.

        Args:
            log_key_dict_file_path (str): file path of the file

        Returns:
            loaded_data (dict): dictionary containing log keys
        '''
        if not os.path.exists(log_key_dict_file_path):
            print(f"File '{log_key_dict_file_path}' does not exist. Creating a new file.")
            with open(log_key_dict_file_path, "w") as new_file:
                new_file.write('{}')  # Write an empty JSON object to the new file

        with open(log_key_dict_file_path, "r") as json_file:
            loaded_data = json.load(json_file)

            # Check if loaded_data is a dictionary
            if not isinstance(loaded_data, dict):
                print("Loaded data is not a dictionary.")

        return loaded_data


    def save_temp_log_key(self, log_key_dict_file_path):

        '''
        Function that saves the temporary log key to file

        Args:
            log_key_dict_file_path (str): file path of the file

        Returns:
            current_index (int): index of the current log key
        '''

        temp = self.temporary_log_key
        temp_dict = self.get_log_key_dict(log_key_dict_file_path)

        # new index is computed as max of old indices + 1
        if not len(temp_dict.items()):
            # add new value
            temp_dict[temp] = 0
        else:
            for k, v in temp_dict.items():
                if temp not in temp_dict.keys():
                    new_val = max(temp_dict.values()) + 1
                    #print(new_val)
                    temp_dict.update({temp:new_val})
                    break
        
        current_index = temp_dict[temp]

        # dictionary is saved to file
        with open(log_key_dict_file_path, 'w') as json_file:
            json.dump(temp_dict, json_file)
        
        return current_index