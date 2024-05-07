import re
import pandas as pd

def preprocess_messages(df):
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
    return df