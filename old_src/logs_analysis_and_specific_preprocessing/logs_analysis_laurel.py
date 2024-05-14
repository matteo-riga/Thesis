import numpy as np
import re

import parser_1, parser_2
import preprocess_messages_laurel

file_path = 'sample_logs_laurel.txt'
df = parser_1.parse_file_to_df(file_path)

severity_dict = {
    "emerg": 0,
    "alert": 1,
    "crit": 2,
    "error": 3,
    "err":3,
    "warn": 4,
    "notice": 5,
    "info": 6,
    "debug": 7
}

facility_dict = {
    "kern": 0,
    "user": 1,
    "mail": 2,
    "system": 3,
    "auth": 4,
    "<syslog?>": 5,
    "<line printer?>": 6,
    "news": 7,
    "uucp": 8,
    "clock": 9,
    "authpriv": 10,
    "ftp": 11,
    "ntp": 12,
    "<log audit?>": 13,
    "<log alert?>": 14,
    "local0": 16,
    "local1": 17,
    "local2": 18,
    "local3": 19,
    "local4": 20,
    "local5": 21,
    "local6": 22,
    "local7": 23,
    "cron":16, # ipotizzando che cron stia usando local0
    "daemon":17 # ipotizzando che laurel stia usando local1
}

severity_numbers = [severity_dict[elem] for elem in df['severity']]
facility_numbers = [facility_dict[elem] for elem in df['facility']]

df['severity_numbers'] = severity_numbers
df['facility_numbers'] = facility_numbers


#======================================================================================================
# CHECK IDEA
# put severity scores in df to be able to spot anomalies
# choose a function that has very high outliers and very low normal values
# for example an exp function
# An advantage is that we capture the magitude of the anomalies and the information about their scale
# of importance
normal_level = "info"
normal_level_numerical = severity_dict[normal_level]
# this way the normal level severity score is 1
severity_scores = [np.exp(normal_level_numerical-elem) for elem in range(0,8)]
#print(severity_scores)
#print(severity_scores[severity_dict['emerg']])  # severity score for emergency level

df['severity_scores'] = [np.exp(normal_level_numerical-elem) for elem in df['severity_numbers']]
# This strategy leverages the domain specific knowledge that some severities are worse than others
#======================================================================================================

print(df.head())

#======================================================================================================
# Now we analyse the message column
# we are doing [27:] to strip out the timestamp. Need to find a better way to do this
# we also need to add a { at the beginning
complex_id_pattern = r'\d+\.\d+:\d+'
line = re.sub(r'"\d+\.\d+:\d+",', '{', df['message'][0], count=1)  #  NB we substitute only the first occurrence
#print(line)

parsed = parser_2.parse_line(line)
attributes = list(parsed.keys())
#print(attributes)
'''
print("parent keys")
print(parsed.keys())
print('======================')
#print(parsed['SYSCALL'].keys()) # here is all the information about uid, euid, suid etc...

print("child keys")
for key in attributes:
    try:
        print(parsed[key].keys())
    except:
        print(key)
'''

keynames = []
keyvalues = []
orig_key = ''

def extract_structure(parsed, key, Q, K):
    #print(f"current key : {key}")
    try:
        keys = list(parsed.keys())
        #print(keys)
        for key in keys:
            extract_structure(parsed[key] ,key, Q, K)
    except:
        #print(parsed)
        Q.append(key)
        K.append(parsed)
        #return parsed
    
extract_structure(parsed, orig_key, keynames, keyvalues)



general_dict = {}
for i in range(len(keynames)):
    general_dict.update({keynames[i]:keyvalues[i]})
    
print(general_dict['PATH'][0])
print(general_dict['PATH'][1])

# From the decomposition of this file we now have, inside general_dict, all the parameters that we need to proceed
# For example we can check the SUID bit, or if the UID and EUID correspond, or things like that
# Now we have to define which of these things we want to keep track of.
# 1. SUID bit
# df['suid'] = general_dict['suid']



#======================================================================================================

# Now we create a minimal df with only numerical values
# here log key approach is unfeasible
# big number of parameters, see which ones we need
# time colunm must be dropped
# columns host, ident, pid, ip, port, user must be one-hot encoded