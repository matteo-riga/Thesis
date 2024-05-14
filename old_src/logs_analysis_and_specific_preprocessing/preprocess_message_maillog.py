import find_sender_receiver, pair_messages

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
    return mail_msg_df