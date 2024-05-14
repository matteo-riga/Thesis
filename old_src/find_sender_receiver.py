import re

def find_sender(text):
    pattern = r'from=<([^<>]+)>'
    matches = re.findall(pattern, text)
    return matches[0] if len(matches)>0 else None

def find_receiver(text):
    pattern = r' to=<([^<>]+)>'
    matches = re.findall(pattern, text)
    return matches[0] if len(matches)>0 else None

def find_alphanumeric_code(text):
    pattern = r'\b[A-F0-9]{12}\b'
    matches = re.findall(pattern, text)
    return matches[0] if len(matches)>0 else None

def find_status(text):
    pattern = r'(?<=status=).*(?=)'
    matches = re.findall(pattern,text)
    return matches[0] if len(matches)>0 else None
