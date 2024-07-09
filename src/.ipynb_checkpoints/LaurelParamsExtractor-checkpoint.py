import os, sys, re, pandas as pd

class LaurelParamsExtractor():
    def __init__(self):
        pass

    def get_params(self, df):
        data_list = []
        for i, line in enumerate(df['message']):
            params = self.get_params_line(line)
            data_list.append(params)

        df = pd.DataFrame(data_list)
        df.columns = ['suid', 'comm', 'exit']
        return df

    def find_pattern(self, pattern, text):
        '''
        Function that finds a regex pattern in a log line.

        Args:
            pattern (r'str): the regex pattern to be found
            text (str): the log line
            sub_pattern(r'str): the regex pattern to be substituted

        Returns:
            result (str/int): the first match found or the value -1 if
                                nothing is found
        '''
        matches = re.findall(pattern, text)
        result = matches[0] if len(matches)>0 else -1
        return result

    def get_params_line(self, line):
        suid_pattern = r' suid=([^\s]+)'
        comm_pattern = r' comm=([^\s]+)'
        exit_pattern = r' exit=([^\s]+)'

        suid = self.find_pattern(suid_pattern, line)
        comm = self.find_pattern(comm_pattern, line)
        exit = self.find_pattern(exit_pattern, line)

        return [suid, comm, exit]