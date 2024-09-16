import re

def parse_audit_log(log: str):
    # Initialize an empty dictionary to hold the key-value pairs
    result = {}

    # Use regular expressions to find all key-value pairs
    # The pattern matches sequences of the form key=value or key="value"
    matches = re.findall(r'(\w+)=("[^"]+"|\S+)', log)

    # Iterate over all matches and add them to the dictionary
    for key, value in matches:
        # Remove quotes from the value if they exist
        result[key] = value.strip('"')

    # columns to keep: cwd/exe, exit, items, ppid, pid, comm, timedelta, pid timedelta, id_anomalies, num_id_anomalies
    
    
    return result