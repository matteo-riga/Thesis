import pandas as pd
import re
import os
import subprocess
import json
import numpy as np
from datetime import datetime

import Reader
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
        self.dang_dir = dangerous_directories.dangerous_directories
        self.dang_dir_outside_cron = dangerous_directories.dangerous_directories_outside_cron
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

        
    def find_file_paths(self, line):
        '''
        Function that finds if there are file paths in a log line.

        Args:
            line (str): the log line

        Returns:
            file_paths (list): the list of found file paths
        '''
        file_path_pattern = r'\/(?:[^\/\s]+\/?)+\b'
        sub_pattern = r'*'
        file_paths = re.findall(file_path_pattern, line)
        self.temporary_log_key = re.sub(file_path_pattern, sub_pattern, self.temporary_log_key)
        return file_paths if len(file_paths) > 0 else -1


    def file_path_length(self, file_path):
        '''
        Function that calculates length of a file path.

        Args:
            line (str): the log line

        Returns:
            num_dir (int): the length of the file path
        '''
        directories = file_path.strip("/").split("/")
        num_dir = len(directories)
        return num_dir


    def find_dangerous_directories(self, line):
        '''
        Function that finds if user is performing a perimeter violation.
        We check if any of the file paths in the log line are from a
        "dangerous" area. Dangerous areas are contained in the imported
        file. They are organized into "dangerous directories" which are
        directories generally dangerous and "dangerous directories outside
        cron" which are directories that are dangerous when we are not
        analysing the cron processes.

        Args:
            line (str): log line that contains the file paths

        Returns:
            n_dang (bool): flag that says if there has been a perimeter
                            violation
            n_dang_outside_cron (bool): flag that says if there has been
                            a perimeter violation when we are outside
                            the cron processes
        '''

        filepaths = self.find_file_paths(line)
        dangerous_paths = [path for path in filepaths if any(path.startswith(directory)) for directory
                           
                           in self.dang_dir]
        dangerous_paths_outside_cron = [path for path in filepaths if any(path.startswith(directory)) for directory in self.dang_dir_outside_cron]
        n_dang = len(dangerous_paths)
        n_dang_outside_cron = len(dangerous_paths)

        return n_dang, n_dang_outside_cron



    def check_dangerous_configuration(self, line):
        '''
        Function that checks if the file configuration is safe.
        Problem is that we do not have access to the file
        '''
        danger_flag = 0
        files = self.find_file_paths(line)
        # Check file permissions for sensitive system files
        sensitive_files = ['/etc/passwd', '/etc/sudoers']
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                permissions = oct(os.stat(file_path).st_mode)[-3:]
                if permissions != '600':
                    danger_flag += 1
                    print(f"Warning: File {file_path} has insecure permissions ({permissions}).")

        return danger_flag



    def find_pattern(self, pattern, text, sub_pattern):
        '''
        Function that finds a regex pattern in a log line and updates
        the current log key by substituting that parameter with a smaller
        general "sub_pattern".

        Args:
            pattern (r'str): the regex pattern to be found
            text (str): the log line
            sub_pattern(r'str): the regex pattern to be substituted

        Returns:
            result (str/int): the first match found or the value -1 if
                                nothing is found
        '''
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        result = matches[0] if len(matches)>0 else -1
        return result

    
    def decode_capabilities(self, cap_fe):
        capabilities = []
        for bit in range(36):  # There are 36 capabilities as of Linux 4.3
            if cap_fe & (1 << bit):
                capabilities.append(capabilities.capabilities_codes[bit])
        return capabilities


    def perform_pattern_substitution(self, line):
        '''
        Performs pattern substitution on the given log line. Runs the find_pattern
        function multiple times to get all the found parameters.

        Args:
            line (str): the log line

        Returns:
            params (list): the list of parameters and the log key number
        '''

        # Find common file paths
        file_paths = self.find_file_paths(line)
        try:
            common_prefix = os.path.commonprefix(file_paths)
            common_directory = os.path.dirname(common_prefix)
            fp_length = self.file_path_length(common_directory)
        except:
            common_prefix, common_directory, fp_length = -1, -1, -1

        # Find and substitute processes
        processes_patterns = [r'Process ([0-9]+)', r'process ([0-9]+)', r' Process ([0-9]+)', r' process ([0-9]+)']
        sub = r'process *'
        processes = [self.find_pattern(p, line, sub) for p in processes_patterns]
        process = processes[0]

        # Find and substitute user
        user_patterns = [r'user ([0-9]+)', r'User ([0-9]+)', r' user ([0-9]+)', r' User ([0-9]+)', r'user ([a-zA-Z0-9]+)', r'User ([a-zA-Z0-9]+)', r' user ([a-zA-Z0-9]+)', r' User ([a-zA-Z0-9]+)']
        sub = r'user *'
        users = [self.find_pattern(p, line, sub) for p in user_patterns]
        user = users[0]

        # Find and substitute sender
        sender_patterns = [r'from=<([^<>]*)>', r' from=<([^<>]*)>']
        sub = r'from=*'
        senders = [self.find_pattern(p, line, sub) for p in sender_patterns]
        sender = senders[0]

        # Find and substitute receiver
        receiver_patterns = [r'to=<([^<>]+)>', r' to=<([^<>]+)>']
        sub = r'to=*'
        receivers = [self.find_pattern(p, line, sub) for p in receiver_patterns]
        receiver = receivers[0]

        # Find and substitute nrcpt
        nrcpt_patterns = [r'nrcpt=[0-9]*', r' nrcpt=[0-9]*']
        sub = r'nrcpt=*'
        nrcpts = [self.find_pattern(p, line, sub) for p in nrcpt_patterns]
        nrcpt = nrcpts[0]

        # Find and substitute size
        size_patterns = [r'size=[0-9]*', r' size=[0-9]*']
        sub = r'size=*'
        sizes = [self.find_pattern(p, line, sub) for p in size_patterns]
        size = sizes[0]

        # Find and substitute alphanumeric codes
        alphan_patterns = [r'\b[A-F0-9]{12}\b']
        sub = r'*'
        alphans = [self.find_pattern(p, line, sub) for p in alphan_patterns]
        alphanum_code = alphans[0]

        # Find and substitute statuses
        statuses_patterns = [r'(?<=status=).*(?=)']
        sub = r'status=*'
        statuses = [self.find_pattern(p, line, sub) for p in statuses_patterns]
        status = statuses[0]

        # Find IP addresses
        ip_patterns = [r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b']
        sub = r'*'
        ips = [self.find_pattern(p, line, sub) for p in ip_patterns]
        ip = ips[0]

        # Find ports
        port_patterns = [r'port ([0-9]+)', r'Port ([0-9]+)', r' port ([0-9]+)', r' Port ([0-9]+)']
        sub = r'port *'
        ports = [self.find_pattern(p, line, sub) for p in port_patterns]
        port = ports[0]

        # Find and substitute sessions
        sessions_patterns = [r'session ([a-zA-Z0-9]+)', r'session-([a-zA-Z0-9]+)', r'Session ([a-zA-Z0-9]+)', r'Session-([a-zA-Z0-9]+)']
        sub = r'session *'
        sessions = [self.find_pattern(p, line, sub) for p in sessions_patterns]
        session = sessions[0]

        # Find and substitute useless parameters
        patterns = [r'Accepted publickey for ([a-zA-Z0-9]+)', r'orig_to=<([^<>]*)>', r'relay=[a-zA-Z]*', r'delay=[0-9]+(.[0-9]+)*', r'delays=[0-9]+(.[0-9]+)*', r'dsn=([0-9].*)+[0-9]', r'message-id=<([^<>]*)>', r'UID [0-9]+', r'file size:  [0-9]+', r'ssh2: [A-Z]+-[A-Z]+', r'SHA256:[A-Za-z0-9]+', r'ID [A-Za-z0-9]+.[A_Za-z0-9]+@[A_Za-z0-9]+.[a-z]+', r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', r'\d{4}-\d{2}-\d{2}']
        subs = [r'Accepted publickey for *'] + ['*' for i in range(len(patterns)-1)]
        useless_params = [self.find_pattern(patterns[i], line, subs[i]) for i in range(len(patterns))]

        # Find and substitute anything useless between parentheses
        pattern = r'\(.*?\)'
        sub_pattern = r'(*)'
        u = self.find_pattern(pattern, line, sub_pattern)

        # Strip unuseful info after dumped core
        pattern = r'dumped core.*'
        sub_pattern = r'dumped core.'
        u = self.find_pattern(pattern, line, sub_pattern)
        

        # Save the final log key
        log_key_file_path = 'log_key.json'
        log_key_n = self.save_temp_log_key(log_key_file_path)

        # Create the final list of parameters
        params = [process, user, sender, receiver, nrcpt, alphanum_code, status, ip, port, session, log_key_n]
        return params


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
                #suid = parsed['SYSCALL']['suid']
                cap_fp = parsed['PATH'][0]['cap_fp'] # extracting binary capabilities
                cap_fp = int(cap_fp, 0)
            except:
                cap_fp = -1

            params = [suid, cap_fp]
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
            params = self.perform_pattern_substitution(line)

        # otherwise
        else:
            self.temporary_log_key = line
            try:
                log_key_number = s.get_log_key(line).get_id()
            except:
                log_key_number = -1
            params = self.perform_pattern_substitution(line)

        return params + [log_key_number]


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
            df.columns = ['process', 'user', 'sender', 'receiver', 'nrcpt', 'alphanum_code', 'status', 'ip', 'port', 'session', 'log key', 'log key spell']
        except:
            # Laurel case
            df.columns = ['suid', 'cap_fp', 'log key']

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
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S %z')
                date_prev = datetime.strptime(self.df['time'][i-1], '%Y-%m-%d %H:%M:%S %z')
                timedelta = date - date_prev
                timedeltas.append(timedelta.total_seconds())

        self.df['timedelta'] = timedeltas

        return self.df