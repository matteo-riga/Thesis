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

def save(df, save_fp):
    # Save this df to file
    #save_fp = '../data/laurel_anomalous_new/save1.csv'
    df.to_csv(save_fp, index=False)

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
    
    df = normalize(df, 'CWD')
    df = normalize(df, 'PATH')
    df = normalize(df, 'SYSCALL')
    df = normalize(df, 'PROCTITLE')
    df = normalize(df, 'EXECVE')
    df = normalize(df, 0)
    df = normalize(df, 1)
    print(df.head())

    columns_to_keep = ['cwd', 'exit', 'items', 'ppid', 'pid', 'comm', 'timedelta', 'pid_timedelta', 'ppid_timedelta', 'id_anomalies', 'num_id_anomalies']
    columns_to_drop = ['ID', 'PID.EVENT_ID', 'PPID.EVENT_ID', 'unix_time', 'ARGV_PROCTITLE', 'time', 'pid_time', 'ppid_time']
    
    try:
        df['ARGV_PROCTITLE_str'] = df['ARGV_PROCTITLE'].apply(lambda x: ' '.join(x))
    except:
        pass
    
    df['unix_time'] = pd.to_numeric(df['ID'].str.split(':').str[0])
    df['timedelta'] = df['unix_time'].diff()
    df['time'] = pd.to_datetime(df['unix_time'], unit='s')

    try:
        df['ppid_time'] = pd.to_datetime(pd.to_numeric(df['PPID.EVENT_ID'].str.split(':').str[0]), unit='s')
        df['ppid_timedelta'] = df['ppid_time'].diff()
        columns_to_keep.remove('ppid_timedelta')
        columns_to_drop.remove('ppid_time')
    except:
        columns_to_drop.remove('ppid_time')
        columns_to_keep.remove('ppid_timedelta')
        columns_to_drop.remove('PPID.EVENT_ID')

    try:
        df['pid_time'] = pd.to_datetime(pd.to_numeric(df['PID.EVENT_ID'].str.split(':').str[0]), unit='s')
        df['pid_timedelta'] = df['pid_time'].diff()
        columns_to_keep.remove('pid_timedelta')
        columns_to_drop.remove('pid_time')
    except:
        columns_to_drop.remove('pid_time')
        columns_to_keep.remove('pid_timedelta')
        columns_to_drop.remove('PID.EVENT_ID')
    
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
            x = pd.Series({col: int(not(row[col] in row['UID_GROUPS'])) for col in columns})
        except:
            x = None
        return x
    
    checks = df.apply(check_id_row, axis=1)
    
    df['id_anomalies'] = checks.sum(axis=1)
    df = df.drop(columns, axis=1)
    df = df.drop('UID_GROUPS', axis=1)

    # filter final columns
    df = df [columns_to_keep]

    return df