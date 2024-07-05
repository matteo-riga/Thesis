import pandas as pd
import re
import os
import subprocess
import json
import numpy as np
from datetime import datetime

import Reader
import PerimeterViolation
import PatternExtractor
import Spell
import SpellLogKeyManager
import severity_codes, facility_codes
import dangerous_directories
import capabilities

import sys
sys.path.append('../../../spell/pyspell')
import spell as s
import pickle


class ParamsExtractor():

    def __init__(self, df):
        self.df = df
        self.temporary_log_key = ''


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


    def get_params_line(self, line):

        '''
        Identifies in which type of log we are and calls the parameter substitution
        accordingly

        Args:
            line (str): the log line

        Returns:
            params (list): the list of extracted parameters specific to the log type
        '''

        s = SpellLogKeyManager.SpellLogKeyManager()
        s.load_spell_model()

        # Understand if we are inside laurel
        complex_id_pattern = r'"\d+\.\d+:\d+",'
        matches = re.findall(complex_id_pattern, line)
        res = len(matches)

        # If we are inside laurel
        if res > 0 or '\"NODE\"' in line:
            r = Reader.Reader('', laurel=1)
            line = re.sub(complex_id_pattern, '{', line, count=1)
            parsed = r.parse_line(line)
            try:
                suid = parsed['SYSCALL']['suid']
            except:
                suid = -1
            
            try:
                comm = parsed['SYSCALL']['comm']
            except:
                comm = -1
                
            try:
                #suid = parsed['SYSCALL']['suid']
                cap_fp = parsed['PATH'][0]['cap_fp'] # extracting binary capabilities
                cap_fp = int(cap_fp, 0)
            except:
                cap_fp = -1

            perimeter = PerimeterViolation.PerimeterViolation().analyze_line(line)
            params = [suid, cap_fp, comm]
            self.temporary_log_key = '*'
            log_key_number = -1

        # If we are inside cron
        elif 'CMD' in line:
            pattern = r'\((.*?)\) ([a-zA-Z]+) \((.*?)\)'
            match = re.match(pattern, line)

            user_cron = match.group(1)
            cmd_cron = match.group(2)
            other_text = match.group(3)
            line = other_text
            self.temporary_log_key = line
            try:
                log_key_number = s.get_log_key(line).get_id()
            except:
                log_key_number = -1
            perimeter = PerimeterViolation.PerimeterViolation().analyze_line(line)
            params = PatternExtractor.PatternExtractor.analyze_line(line)

        # otherwise
        else:
            self.temporary_log_key = line
            try:
                log_key_number = s.get_log_key(line).get_id()
            except:
                log_key_number = -1
            perimeter = PerimeterViolation.PerimeterViolation().analyze_line(line)
            params = PatternExtractor.PatternExtractor().analyze_line(line, self.temporary_log_key)

        return params + [log_key_number] + perimeter


    def get_params(self):
        '''
        Builds the dataframe of extracted parameters from the whole dataframe.
        Applies a mask to eliminate columns with only -1s.

        Args:

        Returns:
            df (pandas DataFrame): the dataframe containing the parameters
        '''
        params_list = []
        for line in self.df['message']:
            params = self.get_params_line(line)
            params_list.append(params)

        #print(params_list)
        df = pd.DataFrame(params_list)
        try:
            df.columns = ['process', 'user', 'sender', 'receiver', 'nrcpt', 'alphanum_code', 'status', 'ip', 'port', 'session', 'log key', 'log key spell', 'n_dang', 'n_dang_no_cron', 'fp_length']
        except:
            # Laurel case
            df.columns = ['suid', 'cap_fp', 'comm', 'log key', 'n_dang', 'n_dang_no_cron', 'fp_length']

        mask = (df != -1).any()
        df = df.loc[:,mask]
        return df


    def convert_params(self, normal_level='info'):
        '''
        Converts severity and facility parameters to numerical 
        values. Computes the time differences and returns them.

        Args:
            normal_level (str): normal level for severity

        Returns:
            self.df (pandas DataFrame): new dataframe
        '''

        severity_numbers = [severity_codes.severity_dict[elem] for elem in self.df['severity']]
        facility_numbers = [facility_codes.facility_dict[elem] for elem in self.df['facility']]

        self.df['severity_numbers'] = severity_numbers
        self.df['facility_numbers'] = facility_numbers

        normal_level = "info"
        normal_level_numerical = severity_codes.severity_dict[normal_level]

        self.df['severity_scores'] = [np.exp(normal_level_numerical-elem) for elem in self.df['severity_numbers']]

        timedeltas = []
        for i, date_str in enumerate(self.df['time']):
            if i == 0:
                timedeltas.append(0)
            else:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S %z')
                    date_prev = datetime.strptime(self.df['time'][i-1], '%Y-%m-%d %H:%M:%S %z')
                    timedelta = date - date_prev
                    timedeltas.append(timedelta.total_seconds())
                except:
                    date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f%z')
                    date_prev = datetime.strptime(self.df['time'][i-1], '%Y-%m-%dT%H:%M:%S.%f%z')
                    timedelta = date - date_prev
                    timedeltas.append(timedelta.total_seconds())

        self.df['timedelta'] = timedeltas

        return self.df