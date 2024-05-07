import re

def find_process(text):
    pattern = r'Process ([0-9]+)'
    matches = re.findall(pattern, text)    
    return matches[0] if len(matches)>0 else None

def find_user_n(text):
    pattern = r'user ([0-9]+)'
    matches = re.findall(pattern,text)
    return matches[0] if len(matches)>0 else None
