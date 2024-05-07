import re
import find_process
import find_log_keys

def preprocess_message(df):
    procs = []
    user_ns = []
    keys = []
    log_keys = find_log_keys.extract_keys_user(df['message'])
    print(log_keys)

    for m in df['message']:
        proc = find_process.find_process(m)
        user_n = find_process.find_user_n(m)
        key = find_log_keys.get_key_from_line_user(m)
        procs.append(proc)
        user_ns.append(user_n)
        keys.append(log_keys[key])

    df['process'] = procs
    df['user_ns'] = user_ns
    df['keys'] = keys

    return df

    