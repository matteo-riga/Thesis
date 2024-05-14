import re
import numpy as np
import pandas as pd
import os

import find_log_keys, find_process, find_sender_receiver, find_file_paths
import find_sender_receiver, pair_messages
import parser_2

# Now for each log report type we need to call the appropriate preprocessing function
# Dataframes are contained in df_list
# Names of log types are containes in log_types

def get_function(log_type):
    switcher = {
        "cron": preprocess_message_cron,
        "laurel": preprocess_message_laurel,
        "maillog": preprocess_message_maillog,
        "messages": preprocess_message_messages,
        "secure": preprocess_message_secure,
        "user": preprocess_message_user
    }

    return switcher.get(log_type, "Invalid Log Type")

def preprocess_message_secure(df):
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
    df = pd.get_dummies(df, columns=['ip', 'port', 'user', 'key'])
    return df


def preprocess_message_messages(df):
    params = []
    for m in df['message']:
        user_pattern = r'user ([a-zA-Z0-9]+)'
        session_id_pattern = r'Session ([a-zA-Z0-9]+)'
        session_id_pattern_2 = r'session ([a-zA-Z0-9]+)'

        matches = re.findall(user_pattern, m)
        user = matches[0] if len(matches) > 0 else None

        matches = re.findall(session_id_pattern, m)
        session_1 = matches[0] if len(matches) > 0 else None

        matches = re.findall(session_id_pattern_2, m)
        session_2 = matches[0] if len(matches) > 0 else None

        if session_1 != None:
            session = session_1
        elif session_2 != None:
            session = session_2
        else:
            session = None

        params.append([user, session])

    params_df = pd.DataFrame(params)
    params_df.columns = ['user', 'session']
    df = pd.concat([df, params_df], axis = 1)
    df = pd.get_dummies(df, columns=['user', 'session'])
    return df

def preprocess_message_user(df):
    procs = []
    user_ns = []
    keys = []
    log_keys = find_log_keys.extract_keys_user(df['message'])
    #print(log_keys)

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

    df = pd.get_dummies(df, columns=['process', 'user_ns', 'keys'])

    return df

def preprocess_message_maillog(df, verbose=False):
    mail_msg = []
    for i, msg in enumerate(df['message']):

        # Dataframe basic attributes
        host = df['host'].iloc[i]
        ident = df['ident'].iloc[i]
        pid = df['pid'].iloc[i]
        severity = df['severity'].iloc[i]
        facility = df['facility'].iloc[i]
        time = df['time'].iloc[i]
        sev_n = df['severity_numbers'].iloc[i]
        fac_n = df['facility_numbers'].iloc[i]
        sev_sc = df['severity_scores'].iloc[i]

        sender = find_sender_receiver.find_sender(msg)
        receiver = find_sender_receiver.find_receiver(msg)
        alphanum_code = find_sender_receiver.find_alphanumeric_code(msg)
        status = find_sender_receiver.find_status(msg)

        '''
        print(f"=========================")
        print(f"Sender: {sender}")
        print(f"Receiver: {receiver}")
        print(f"ID code: {alphanum_code}")
        print(f"Status: {status}")
        '''
        #print(alphanum_code, sender, receiver, status)
        mail_msg.append([alphanum_code, sender, receiver, status, host, ident, pid, time, severity, facility, sev_n, fac_n, sev_sc])

        # ********************************* #
        # IDEA: we can parse the text of the message and try to use a language model OR a sentiment analysis
        #sentiment = sentiment_model.predict(msg)
        
    # Now i have to find a way to pair these messages by alphanumeric code and receiver/sender
    #print(mail_msg[0:10])

    mail_msg_df = pair_messages.pair_messages(mail_msg, verbose=False)

    if verbose:
        print(f"In this message dataframe, with shape {df.shape}, there are:")
        nIDs = len(np.unique(mail_msg_df['ID']))
        nsenders = len(np.unique(mail_msg_df['sender']))
        nreceivers = len(np.unique(mail_msg_df['receiver']))
        nstatuses = len(np.unique(mail_msg_df['status']))
        print(f"{nIDs} unique IDs")
        print(f"{nsenders} unique senders")
        print(f"{nreceivers} unique receivers")
        print(f"{nstatuses} statuses")

        for i in range(10):
            print('=======================================')
            print(mail_msg_df.iloc[i])

    #df = pd.get_dummies(df, columns=[''])
    return mail_msg_df


# CHECK IF IT WORKSs
def preprocess_message_cron(df):

    # Extract user, command, and other text from each message and store in lists
    users = []
    commands = []
    other_texts = []

    messages = df['message']

    for text in messages:
        # Define a regular expression pattern to match the user, command, and other text
        pattern = r'\((.*?)\) ([a-zA-Z]+) \((.*?)\)'

        # Use re.match() to search for the pattern in the text
        match = re.match(pattern, text)

        if match:
            # Extract the user, command, and other text
            user = match.group(1)
            cmd = match.group(2)
            other_text = match.group(3)

            # Add extracted values to lists
            users.append(user)
            commands.append(cmd)
            other_texts.append(other_text)
        else:
            # If no match found, append None to the lists
            users.append(-1)
            commands.append(None)
            other_texts.append(text)

    # Add new columns to the DataFrame
    df['user'] = users
    df['message'] = other_texts

    # Display the updated DataFrame
    #print(df)
    common_file_paths = []
    common_file_paths_size = []

    for message in messages:
        file_paths = find_file_paths.find_file_paths(message)
        common_prefix = os.path.commonprefix(file_paths)
        common_directory = os.path.dirname(common_prefix)
        common_file_paths.append(common_directory)
        common_file_paths_size.append(find_file_paths.file_path_length(common_directory))

    #df['common_file_paths'] = common_file_paths
    df['common_file_paths_length'] = common_file_paths_size

    #print(df.head())

    #df = df.drop(columns=['common_file_paths'])

    df = pd.get_dummies(df, columns=['user'])

    return df

def preprocess_message_laurel(df):
    # REMEMBER TO ADD MORE MEANINGFUL PARAMETERS
    suids = []
    for line in df['message']:
        complex_id_pattern = r'"\d+\.\d+:\d+",'
        line = re.sub(complex_id_pattern, '{', df['message'][0], count=1)
        parsed = parser_2.parse_line(line)
        suid = parsed['SYSCALL']['suid']
        suids.append(suid)

    df['suid'] = suids
    return df