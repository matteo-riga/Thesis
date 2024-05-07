import numpy as np

import parser_1
import preprocess_messages_messages

file_path = 'sample_logs_messages.txt'
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
    "daemon":17 # ipotizzando che cron stia usando local1
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

#print(df.head())

#======================================================================================================
# Now we analyse the message column
df = preprocess_messages_messages.preprocess_messages(df)
print(df.head())
print(df.columns)

#======================================================================================================

# Now we create a minimal df with only numerical values
# SEE IF LOG KEY APPROACH IS GOOD
# time colunm must be dropped
# columns host, ident, pid, ip, port, user must be one-hot encoded