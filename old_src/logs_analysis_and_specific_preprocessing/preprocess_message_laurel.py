import re

import parser_2

def preprocess_message_laurel(df):
    suids = []
    for line in df['message']:
        complex_id_pattern = r'"\d+\.\d+:\d+",'
        line = re.sub(complex_id_pattern, '{', df['message'][0], count=1)
        parsed = parser_2.parse_line(line)
        suid = parsed['SYSCALL'].keys()
        suids.append(suid)

    df['suid'] = suids
    return df
        