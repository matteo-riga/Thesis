import pandas as pd
import re
import os
import json
import numpy as np
from datetime import datetime

import Reader
import Spell
import severity_codes, facility_codes
import dangerous_directories

class ParamsExtractor():
    def __init__(self, df):
        self.df = df
        self.dang_dir = dangerous_directories.dangerous_directories
        self.cron_dang_dir = dangerous_directories.dangerous_directories_outside_cron
        pass

    def find_dangerous_directories(self, line):
        # Regex pattern to find file paths
        file_path_pattern = r'\/(?:[^\/\s]+\/?)+\b'
        # Find all file paths in the log line
        file_paths = re.findall(file_path_pattern, line)
        # Check if any of the file paths are in the list of password directories
        dangerous_paths = [path for path in file_paths if any(path.startswith(dir) for dir in self.dang_dir)]
        dangerous_paths_no_cron = [path for path in file_paths if any(path.startswith(dir) for dir in self.cron_dang_dir)]

        dang = len(dangerous_paths) > 0
        no_cron_dang = len(dangerous_paths_no_cron) > 0
        
        return dang, no_cron_dang # simple flag that says if there has been a perimeter violation
        

    def find_file_paths(self, line):
        file_path_pattern = r'\/(?:[^\/\s]+\/?)+\b'
        sub_pattern = r'*'
        file_paths = re.findall(file_path_pattern, line)
        self.temporary_log_key = re.sub(file_path_pattern, sub_pattern, self.temporary_log_key)
        return file_paths
    
    def file_path_length(self, file_path):
        directories = file_path.strip("/").split("/")
        num_directories = len(directories)
        return num_directories
    
    def find_process(self, text):
        pattern = r'Process ([0-9]+)'
        sub_pattern = r'Process *'
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)    
        return matches[0] if len(matches)>0 else -1

    def find_user_n(self, text):
        pattern = r'user ([0-9]+)'
        sub_pattern = r'user *'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1
    
    def find_sender(self, text):
        pattern = r'from=<([^<>]*)>'
        sub_pattern = r'from=*'
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_receiver(self, text):
        pattern = r' to=<([^<>]+)>'
        sub_pattern = r' to=*'
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_size(self, text):
        pattern = r'size=[0-9]*'
        sub_pattern = r'size=*'
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_nrcpt(self, text):
        pattern = r' nrcpt=[0-9]*'
        sub_pattern = r' nrcpt=*'
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_alphanumeric_code(self, text):
        pattern = r'\b[A-F0-9]{12}\b'
        sub_pattern = r'*'
        matches = re.findall(pattern, text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_status(self, text):
        pattern = r'(?<=status=).*(?=)'
        sub_pattern = r'status=*'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_ip(self, text):
        pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        sub_pattern = r'*'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1
    
    def find_port(self, text):
        pattern = r'port ([0-9]+)'
        sub_pattern = r'port *'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1
    
    def find_user_1(self, text):
        pattern = r'Accepted publickey for ([a-zA-Z0-9]+)'
        sub_pattern = r'Accepted publickey for *'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1
    
    def find_user_2(self, text):
        pattern = r'user ([a-zA-Z0-9]+)'
        sub_pattern = r'user *'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_session(self, text):
        patterns = [r'session ([a-zA-Z0-9]+)', r'session-([a-zA-Z0-9]+)', r'Session ([a-zA-Z0-9]+)', r'Session-([a-zA-Z0-9]+)']
        sub_pattern = r'session *'
        for pattern in patterns:
            matches = re.findall(pattern,text)
            self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_useless_params(self, text):
        # orig_to=<a07cmc01>, relay=local, delay=0.98, delays=0.97*, dsn=5.2.2, status=status=*
        patterns = [r'orig_to=<([^<>]*)>', r'relay=[a-zA-Z]*', r'delay=[0-9]+(.[0-9]+)*', r'delays=[0-9]+(.[0-9]+)*', r'dsn=([0-9].*)+[0-9]', r'message-id=<([^<>]*)>', r'UID [0-9]+', r'file size:  [0-9]+', r'ssh2: [A-Z]+-[A-Z]+', r'SHA256:[A-Za-z0-9]+', r'ID [A-Za-z0-9]+.[A_Za-z0-9]+@[A_Za-z0-9]+.[a-z]+']
        sub_pattern = r'*'
        for pattern in patterns:
            matches = re.findall(pattern,text)
            self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def find_between_parenthesis(self, text):
        pattern = r'\(.*?\)'
        sub_pattern = r'(*)'
        matches = re.findall(pattern,text)
        self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)
        return matches[0] if len(matches)>0 else -1

    def get_log_key_dict(self, log_key_dict_file_path):

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

        #self.spell_log_key = line
        #self.process_spell_log_key()

        # We initialize the params to be -1
        user_cron = -1
        fp_length = -1
        process = -1
        user = -1
        sender = -1
        receiver = -1
        alphanum_code = -1
        status = -1
        log_key_n = -1
        ip = -1
        port = -1
        user_1 = -1
        user_2 = -1
        suid = -1
        bet_par = -1

        # Here we treat the line of the df['message'] as a string
        # This assumption holds for every type of log apart from
        # laurel.
        # for laurel we need to implement a specific parsing
        # method.
        
        # ==========================================
        # LAUREL
        # reader initialized with empty filepath since we don't need it
        
        laurel = 0

        #print(line)

        complex_id_pattern = r'"\d+\.\d+:\d+",'
        matches = re.findall(complex_id_pattern, line)
        res = len(matches)

        if res > 0 or '\"NODE\"' in line:
            laurel = 1

        if laurel == 1:
            r = Reader.Reader('', laurel=1)
            line = re.sub(complex_id_pattern, '{', line, count=1)
            parsed = r.parse_line(line)
            
            try:
                suid = parsed['SYSCALL']['suid']
                cap_fp = parsed['PATH'][0]['cap_fp'] # extracting binary capabilities
                if type(cap_fp)==str:
                    cap_fp=-1
                else:
                    cap_fp=int(cap_fp)
            except:
                suid = -1

            self.temporary_log_key = '*'
            
            # remember to write an exit to the function here
            # since we are in the laurel case we don't need the other parameters
            # we use the laurel flag


        else:
            # CRON
            # Here every message line is assumed to be directly a string
            # this is wrong in the case of cron, since in the message
            # variable of cron there is something like:
            # (acctdata) CMD (...)
            # so we have to handle this special case
            if 'CMD' in line:
                pattern = r'\((.*?)\) ([a-zA-Z]+) \((.*?)\)'
                match = re.match(pattern, line)

                user_cron = match.group(1)
                cmd_cron = match.group(2)
                other_text = match.group(3)

                line = other_text

        
            self.temporary_log_key = line

            # ==========================================
            # Find common file paths
            file_paths = self.find_file_paths(line)

            # If no file path is extracted from the line
            # we return the value -1 for the length of the common
            # file path
            if not file_paths:
                fp_length = -1
            else:
                common_prefix = os.path.commonprefix(file_paths)
                common_directory = os.path.dirname(common_prefix)
                fp_length = self.file_path_length(common_directory)
                #print(f"The longest common directory path is: {common_directory}")
                #print(f"And it's long {fp_length} directories")


            # ==========================================
            # Find process and user
            process = self.find_process(line)
            user = self.find_user_n(line)

            # ==========================================
            # Find sender, receiver, status and alphanum_code
            # Improve maillog preprocessing
            sender = self.find_sender(line)
            receiver = self.find_receiver(line)
            size = self.find_size(line)
            nrcpt = self.find_nrcpt(line)
            alphanum_code = self.find_alphanumeric_code(line)
            status = self.find_status(line)

            # ==========================================
            # Find ip, port, user 1 and user 2
            ip = self.find_ip(line)
            port = self.find_port(line)
            user_1 = self.find_user_1(line)
            user_2 = self.find_user_2(line)
            
            # ==========================================
            # Find session
            session = self.find_session(line)

            useless_params = self.find_useless_params(line)

            # ==========================================
            # Find user or parameters between parenthesis
            bet_par = self.find_between_parenthesis(line)

            # cut user python output
            pattern = r'#(.*)'
            sub_pattern = r''
            self.temporary_log_key = re.sub(pattern, sub_pattern, self.temporary_log_key)

            # save self.temporary_log_key to file
            # or add it to dictionary
            log_key_file_path = 'log_key.json'
            log_key_n = self.save_temp_log_key(log_key_file_path)

        return user_cron, \
            fp_length, process, user, \
            sender, receiver, alphanum_code, status, log_key_n, \
            ip, port, user_1, user_2, suid, bet_par

    def get_params(self):
        params_list = []
        for line in self.df['message']:
            params = self.get_params_line(line)
            params_list.append(params)

        df = pd.DataFrame(params_list)
        df.columns = ['user_cron', 'fp_length', 'process', 'user', 'sender', 'receiver', 'alphanum_code', 'status', 'log_key_n', 'ip', 'port', 'user_1', 'user_2', 'suid', 'bet_par']

        # Remove from df columns with all -1s
        mask = (df != -1).any()
        df = df.loc[:, mask]
        return df

    # This function converts the severity and facility parameters to numerical values
    def convert_params(self, normal_level='info'):

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