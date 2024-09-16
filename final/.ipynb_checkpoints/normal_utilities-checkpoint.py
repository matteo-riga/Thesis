import json
import re

def read(fp):
    data_list = []
    lines_limit = 100000 # limit reading lines, done for memory constraints
    i = 0
    
    with open(fp, 'r') as file:
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
                    
    df = pd.DataFrame(data_list)
    return df


def normalize(df, col):
    if col not in df.columns:
        print(f"Error, column {col} not in df")
        return None
    df_expanded = pd.json_normalize(df[col])
    #print(df_expanded.head())
    df = df.drop(col, axis=1)
    df = df.join(df_expanded, rsuffix=f'_{col}')
    return df



# Final Preprocessing Function
def preprocess_laurel(df):

    columns_to_keep = ['cwd', 'exit', 'items', 'ppid', 'pid', 'comm', 'timedelta', 'pid_timedelta', 'ppid_timedelta', 'id_anomalies', 'num_id_anomalies', 'ARGV_PROCTITLE_str']
    columns_to_drop = ['unix_time', 'time', 'pid_time', 'ppid_time']
    
    try:
        df['ARGV_PROCTITLE_str'] = df['ARGV_PROCTITLE'].apply(lambda x: ' '.join(x))
    except:
        columns_to_keep.remove('ARGV_PROCTITLE_str')
        pass

    try:
        time_serie = df['unix_time']
        df['unix_time'] = pd.to_numeric(time_serie.str.split(':').str[0])
        df['timedelta'] = df['unix_time'].diff()
        df['time'] = pd.to_datetime(df['unix_time'], unit='s')
    except:
        pass

    try:
        df['ppid_time'] = pd.to_datetime(pd.to_numeric(df['PPID.EVENT_ID'].str.split(':').str[0]), unit='s')
        df['ppid_timedelta'] = df['ppid_time'].diff()
        columns_to_keep.remove('ppid_timedelta')
        columns_to_drop.remove('ppid_time')
    except:
        columns_to_keep.remove('ppid_timedelta')
        columns_to_drop.remove('ppid_time')
        pass

    try:
        df['pid_time'] = pd.to_datetime(pd.to_numeric(df['PID.EVENT_ID'].str.split(':').str[0]), unit='s')
        df['pid_timedelta'] = df['pid_time'].diff()
        columns_to_keep.remove('pid_timedelta')
        columns_to_drop.remove('pid_time')
    except:
        columns_to_keep.remove('pid_timedelta')
        columns_to_drop.remove('pid_time')
        pass
    
    df = df.drop(columns_to_drop, axis=1)

    columns = ['auid', 'uid', 'gid', 'euid', 'suid', 'fsuid', 'egid', 'sgid', 'fsgid']
    
    def check_row_id_num(row):
        return pd.Series({col: row[col] == 0 for col in columns})
    
    checks = df.apply(check_row_id_num, axis=1)
    df['num_id_anomalies'] = checks.sum(axis=1)
    df = df.drop(columns, axis=1)

    columns = ['AUID', 'UID', 'GID', 'EUID', 'SUID', 'FSUID', 'EGID', 'SGID', 'FSGID']
    
    # Use apply to efficiently check values for each row
    def check_id_row(row):
        try:
            return pd.Series({col: int(not(row[col] in row['UID_GROUPS'])) for col in columns})
        except:
            return None
    
    checks = df.apply(check_id_row, axis=1)
    
    df['id_anomalies'] = checks.sum()
    df = df.drop(columns, axis=1)
    try:
        df = df.drop('UID_GROUPS', axis=1)
    except:
        pass

    # filter final columns
    df = df[columns_to_keep]

    return df





def parse_audit_log(log):
    # Initialize an empty dictionary to hold the key-value pairs
    result = {}

    # Placeholder for timedelta, pid timedelta, and anomaly-related logic
    result['id_anomalies'] = 0  # Placeholder, implement anomaly logic
    result['num_id_anomalies'] = 0  # Placeholder, implement anomaly counting
    
    # Clean the log string by removing unnecessary quotes at the beginning and end
    # and replacing any non-standard characters like '\x1d'
    timestamp_match = re.match(r'"(\d+\.\d+:\d+)",', log)
    if timestamp_match:
        # Extract and store the timestamp in the result dictionary
        result['unix_time'] = timestamp_match.group(1)
    else:
        result['unix_time'] = None
        
    clean_log = re.sub(r'"\d+\.\d+:\d+",', '{', log)
    clean_log += '}'
    #clean_log = log.strip('"')
    #print(clean_log)
    #print('===========================0')

    try:
        # Convert the cleaned JSON log to a dictionary
        data = json.loads(clean_log)

        # Extract relevant fields
        syscall = data.get('SYSCALL', {})
        result['cwd'] = data.get('CWD', {}).get('cwd')
        result['exe'] = syscall.get('exe')
        result['exit'] = syscall.get('exit')
        result['items'] = syscall.get('items')
        result['ppid'] = syscall.get('ppid')
        result['pid'] = syscall.get('pid')
        result['comm'] = syscall.get('comm')

        for str_id in ['auid', 'uid', 'gid', 'euid', 'suid', 'fsuid', 'egid', 'sgid', 'fsgid', 'AUID', 'UID', 'GID', 'EUID', 'SUID', 'FSUID', 'EGID', 'SGID', 'FSGID']:
            result[str_id] = syscall.get(str_id)

    except json.JSONDecodeError as e:
        pass
        #print(f"Error: Couldn't parse the JSON part of the log. {e}")
    
    return result