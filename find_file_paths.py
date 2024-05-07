import re

def find_file_paths(message):
    # Regular expression to match file paths
    file_path_regex = r'\/(?:[^\/\s]+\/?)+\b'
    
    # Find all matches of file paths in the message
    file_paths = re.findall(file_path_regex, message)
    
    return file_paths

def file_path_length(file_path):
    # Split the file path into individual directories
    directories = file_path.strip("/").split("/")
    # Count the number of directories
    num_directories = len(directories)
    return num_directories