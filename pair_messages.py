import pandas as pd

def pair_messages(mail_msg, verbose=False):
    mail_msg_dict = []
    analyzed_ids = []
    for i,msg in enumerate(mail_msg):
        first_ID = msg[0]
        attrs = msg[4:]
        if first_ID not in analyzed_ids:   
            sender = msg[1]
            receiver = msg[2]
            status = msg[3]
            if verbose:
                print("========================")
                print(f"message i {i}")
                print(first_ID, sender, receiver, status)
                if sender == None and i == 0:
                    print(f"Sender to be found in earlier messages")
            else:
                for j,other_msg in enumerate(mail_msg[i + 1:]):
                    if verbose:
                        print(f"message j {j}")
                        print(other_msg[0])
                    if first_ID == other_msg[0]:
                        #print("found same msg ID")
                        receiver = other_msg[2] if other_msg[2] != None else receiver
                        status = other_msg[3] if other_msg[3] != None else status
                        # Now we analyze the attributes field
                        other_attrs = other_msg[4:]
                        if attrs[6] < other_attrs[6]:  # keep highest severity observed
                            attrs[6] = other_attrs[6]
                            attrs[-1] = other_attrs[-1] # update severity score
                if verbose:
                    print(first_ID, sender, receiver, status)
                if sender != None and receiver != None and status != None:
                    mail_msg_dict.append([first_ID, sender, receiver, status] + attrs)
                    analyzed_ids.append(first_ID)
    df = pd.DataFrame(mail_msg_dict)
    df.columns = ['ID', 'sender', 'receiver', 'status', 'host', 'ident', 'pid', 'time', 'severity', 'facility', 'severity_numbers', 'facility_numbers', 'severity_scores']
    return df