import pandas as pd
import re
import os
import subprocess
import json
import numpy as np
from datetime import datetime

import Reader
import dangerous_directories

class PerimeterViolation():

    def __init__(self):
        self.dang_dir = dangerous_directories.dangerous_directories
        self.dang_dir_outside_cron = dangerous_directories.dangerous_directories_outside_cron
        pass


    def find_file_paths(self, line):
        '''
        Function that finds if there are file paths in a log line.

        Args:
            line (str): the log line

        Returns:
            file_paths (list): the list of found file paths
        '''
        file_path_pattern = r'\/(?:[a-zA-Z0-9_\-\.]+\/?)*'
        sub_pattern = r'*'
        file_paths = re.findall(file_path_pattern, line)
        #self.temporary_log_key = re.sub(file_path_pattern, sub_pattern, self.temporary_log_key)
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
        if filepaths != -1:
            dangerous_paths = [path for path in filepaths if any(path.startswith(directory) for directory in self.dang_dir)]
            dangerous_paths_outside_cron = [path for path in filepaths if any(path.startswith(directory) for directory in self.dang_dir_outside_cron)]
            n_dang = len(dangerous_paths)
            n_dang_outside_cron = len(dangerous_paths_outside_cron)
        else:
            n_dang = -1
            n_dang_outside_cron = -1
    
        return n_dang, n_dang_outside_cron


    

    def analyze_line(self, line):
        '''
        Function that finds anomalies in a log line by checking
        for perimeter violations.

        Args:
            line (str): log line

        Returns:
            n (bool): flag that says if there has been a perimeter
                            violation
            n1 (bool): flag that says if there has been
                            a perimeter violation when we are outside
                            the cron processes
            l (int): length of file paths inside the log line
        '''
        n, n1 = self.find_dangerous_directories(line)
        l = self.file_path_length(line)
        return [n, n1, l]