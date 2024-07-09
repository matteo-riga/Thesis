import json
import pandas as pd

class EncodeCommand:
    def __init__(self, file_store='../settings/command_store.json'):
        self.command_dict = {}
        self.file_store = file_store
    
    def encode_command(self, df):
        # List to store the encoded commands
        encoded_commands = []
        
        # Variable to keep track of the next index
        current_index = 1
        
        # Iterate over the DataFrame and populate the command dictionary
        for comm in df['comm']:
            if comm not in self.command_dict.values():
                self.command_dict.update({current_index: comm})
                encoded_commands.append(current_index)
                current_index += 1
            else:
                # Find the existing key for this command
                existing_key = next(key for key, value in self.command_dict.items() if value == comm)
                encoded_commands.append(existing_key)
        
        # Add the list of encoded commands as a new column to the DataFrame
        df['enc_comm'] = encoded_commands
        
        # Save the command dictionary to a file
        with open(self.file_store, 'w') as f:
            json.dump(self.command_dict, f)
        
        return df
    
    def get_command_number(self, command):
        # Load the command dictionary from the file
        with open(self.file_store, 'r') as f:
            command_dict = json.load(f)
        
        # Search for the command in the dictionary and return the index
        for key, value in command_dict.items():
            if value == command:
                return key
        return None

# Example usage
# Create an instance of EncodeCommand
# encoder = EncodeCommand()

# Assuming df is your DataFrame with a 'comm' column
# df = pd.DataFrame({'comm': ['ping', 'ping', 'traceroute', 'ping']})

# Encode the commands and get the updated DataFrame
# df_encoded = encoder.encode_command(df)
# print(df_encoded)

# Get the command number
# command_number = encoder.get_command_number('ping')
# print(command_number)  # Output should be 1 if 'ping' is the first command
