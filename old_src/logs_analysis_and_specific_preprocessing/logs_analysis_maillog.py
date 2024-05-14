import numpy as np
import pandas as pd
import os

import parser_1
import find_sender_receiver
import pair_messages
import preprocess_message_maillog

file_path = 'sample_logs_maillog_extended.txt'
df = parser_1.parse_file_to_df(file_path)

severity_dict = {
    "emerg": 0,
    "alert": 1,
    "crit": 2,
    "error": 3,
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
    "cron":16 # ipotizzando che cron stia usando local0
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

df['severity_scores'] = [np.exp(normal_level_numerical - elem) for elem in df['severity_numbers']]
# This strategy leverages the domain specific knowledge that some severities are worse than others
#======================================================================================================

print(df.columns)

#======================================================================================================
# Now we analyse the message column
# We have to extract sender and receiver for every message

mail_msg_df = preprocess_message_maillog.preprocess_message_maillog(df, verbose=False)
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

#======================================================================================================

# Now we create a minimal df with only numerical values
# At this point the ID column is not useful anymore, can be dropped
mail_msg_df = mail_msg_df.drop(columns = 'ID')
# Other columns such as senders, receivers and statuses will be one-hot encoded
categorical_columns = ['sender', 'receiver', 'status']
mail_msg_df = pd.get_dummies(mail_msg_df, columns=categorical_columns, drop_first=True)

# NOW FIND A WAY TO MERGE IT WITH OTHER INFORMATIONS IN THE df
# done, can be improved
# up until now the new dataframe contains all the df information of the FIRST message
# with a given ID. for example, it contains the pid of the first message (sender process)
# BUT with the slight improvement that it keeps the highest recorded severity number and
# severity score.
# This can be improved by adding as parameter the list of attributes of all the messages
# and by adding the log key