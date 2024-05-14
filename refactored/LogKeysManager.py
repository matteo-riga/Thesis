import json
import os

class LogKeysManager():
    def __init__(self, file_path):
        self.file_path = file_path

    def get_log_keys(self):

        if not os.path.exists(self.file_path):
            print(f"File '{self.file_path}' does not exist. Creating a new file.")
            with open(self.file_path, "w") as new_file:
                new_file.write('{}')  # Write an empty JSON object to the new file

        with open(self.file_path, "r") as json_file:
            loaded_data = json.load(json_file)

            # Check if loaded_data is a dictionary
            if not isinstance(loaded_data, dict):
                print("Loaded data is not a dictionary.")

        return loaded_data