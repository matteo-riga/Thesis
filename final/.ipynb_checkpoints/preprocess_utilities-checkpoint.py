import hashlib

def count_directories(path):
    try:
        # Split the path into components
        components = path.split(os.sep)
        # Filter out empty components (which can occur if the path starts with a separator)
        directories = [comp for comp in components if comp]
    except:
        directories = []
    return len(directories)


def string_to_milliseconds(time_str):
    try:
        # Step 1: Parse the string into days and time component
        time_parts = time_str.split(' days ')
        days = int(time_parts[0])
        time_component = time_parts[1]
        
        # Step 2: Split the time component into hours, minutes, seconds, and microseconds
        hours, minutes, seconds = map(float, time_component.replace(':', ' ').split())
        
        # Step 3: Create a timedelta object
        time_delta = timedelta(
            days=days,
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds),
            microseconds=int((seconds - int(seconds)) * 1_000_000)
        )
        
        # Step 4: Convert the timedelta object to milliseconds
        milliseconds = time_delta.total_seconds() * 1000
    except:
        milliseconds = 0
    return abs(milliseconds)


def encode_pid_to_8bit_binary_hash(pid):
    """
    Encode a PID (Process ID) using SHA-256 hash and return an 8-bit binary string.

    Parameters:
    pid (int): The process ID to encode.

    Returns:
    str: An 8-bit binary representation of the hash of the PID.
    """
    # Convert the PID to a string and then to bytes
    pid_bytes = str(pid).encode('utf-8')
    
    # Compute the SHA-256 hash of the PID
    hash_object = hashlib.sha256(pid_bytes)
    
    # Get the hash as a hexadecimal string
    hash_hex = hash_object.hexdigest()
    
    # Convert the first byte (2 hex digits) of the hash to an integer
    first_byte = int(hash_hex[:2], 16)
    
    # Convert the integer to an 8-bit binary string
    binary_hash = format(first_byte, '08b')

    # final int
    r = int(binary_hash,2)
    
    return float(r/256)



def preprocess(df, encoder):
    
    # Select only common 
    cols = ['cwd', 'exit', 'items', 'ppid', 'pid', 'comm', 'timedelta', 'id_anomalies', 'num_id_anomalies']
    df = df[cols]

    # convert times to numbers
    try:
        df['pid_timedelta_ms'] = df['pid_timedelta'].apply(string_to_milliseconds)
        df = df.drop('pid_timedelta', axis=1)
    except:
        pass
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # preprocess cwd column by extracting path length
    df['fp_length'] = df['cwd'].apply(count_directories)
    df = df.drop('cwd', axis=1)

    # encode the command
    df = encoder.encode_command(df)
    df = df.drop('comm', axis=1)

    # encode the pids
    df['pid'] = df['pid'].apply(encode_pid_to_8bit_binary_hash)
    df['ppid'] = df['ppid'].apply(encode_pid_to_8bit_binary_hash)
    

    # convert all df to floats
    df = df.astype(float)
    
    return df
    