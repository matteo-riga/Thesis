import re

def extract_keys_user(msg_list):
    keys = {}
    i = 0
    for m in msg_list:
        pattern_proc = r'Process ([0-9]+)'
        pattern_user = r'user ([0-9]+)'
        substituted_message = re.sub(pattern_proc, r'Process *', m)
        substituted_message = re.sub(pattern_user, r'user *', substituted_message)
        if substituted_message not in keys:
            keys.update({substituted_message:i})
            i += 1
    #log_standard_message = 'Process * (node) of user * dumped core.'
    #log_key = {log_standard_message:1}
    #print(keys)
    return keys

def get_key_from_line_user(m):
    pattern_proc = r'Process ([0-9]+)'
    pattern_user = r'user ([0-9]+)'
    substituted_message = re.sub(pattern_proc, r'Process *', m)
    substituted_message = re.sub(pattern_user, r'user *', substituted_message)
    #print(substituted_message)
    return substituted_message


def extract_keys_secure(msg_list):

    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    port_pattern = r'port ([0-9]+)'
    user_pattern = r'Accepted publickey for ([a-zA-Z0-9]+)'
    user_pattern_2 = r'user ([a-zA-Z0-9]+)'
    patterns = [ip_pattern, port_pattern, user_pattern, user_pattern_2]
    keys = {}
    i = 0

    for m in msg_list:
        substituted_message = m
        for pattern in patterns:
            substituted_message = re.sub(pattern, r'*', substituted_message)
        if substituted_message not in keys:
            keys.update({substituted_message:i})
            i += 1
    
    #print(keys)
    return keys

def get_key_from_line_secure(m):
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    port_pattern = r'port ([0-9]+)'
    user_pattern = r'Accepted publickey for ([a-zA-Z0-9]+)'
    user_pattern_2 = r'user ([a-zA-Z0-9]+)'
    patterns = [ip_pattern, port_pattern, user_pattern, user_pattern_2]

    substituted_message = m
    for pattern in patterns:
        substituted_message = re.sub(pattern, r'*', substituted_message)
    #print(substituted_message)
    return substituted_message