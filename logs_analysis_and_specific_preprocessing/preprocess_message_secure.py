import re
import pandas as pd
import find_log_keys

def preprocess_message(df):
    params = []
    log_keys = find_log_keys.extract_keys_secure(df['message'])

    for m in df['message']:

        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        matches = re.findall(ip_pattern, m)
        ip = matches[0] if len(matches) > 0 else None

        port_pattern = r'port ([0-9]+)'
        matches = re.findall(port_pattern, m)
        port = matches[0] if len(matches) > 0 else None

        user_pattern = r'Accepted publickey for ([a-zA-Z0-9]+)'
        matches = re.findall(user_pattern, m)
        user_accepted = matches[0] if len(matches) > 0 else None

        user_pattern = r'user ([a-zA-Z0-9]+)'
        matches = re.findall(user_pattern, m)
        user_open = matches[0] if len(matches) > 0 else None

        key = find_log_keys.get_key_from_line_secure(m)


        if user_accepted != None:
            user = user_accepted
        elif user_open != None:
            user = user_open
        else:
            user = None

        # maybe there is a more intelligent way to do this
        # for example by filtering only for the user and then putting the information
        # of connection/disconnection in the log key
        params.append([ip, port, user, log_keys[key]])

        #=====================================================================#
        # LOG KEY CREATION
        '''
        substituted_message = re.sub(ip_pattern, r'1.1.1.1', m)
        substituted_message = re.sub(port_pattern, r'0000', substituted_message)
        log_standard_message = 'Process * (node) of user * dumped core.'
        log_key = {log_standard_message:1}
        '''
        #=====================================================================#

    params_df = pd.DataFrame(params)
    params_df.columns = ['ip', 'port', 'user', 'key']
    df = pd.concat([df, params_df], axis=1)
    return df
