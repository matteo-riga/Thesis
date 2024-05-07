import re
import pandas as pd


def preprocess_message(df):

    # Extract user, command, and other text from each message and store in lists
    users = []
    commands = []
    other_texts = []

    messages = df['message']

    for text in messages:
        # Define a regular expression pattern to match the user, command, and other text
        pattern = r'\((.*?)\) ([a-zA-Z]+) \((.*?)\)'

        # Use re.match() to search for the pattern in the text
        match = re.match(pattern, text)

        if match:
            # Extract the user, command, and other text
            user = match.group(1)
            cmd = match.group(2)
            other_text = match.group(3)

            # Add extracted values to lists
            users.append(user)
            commands.append(cmd)
            other_texts.append(other_text)
        else:
            # If no match found, append None to the lists
            users.append(-1)
            commands.append(None)
            other_texts.append(text)

    # Add new columns to the DataFrame
    df['user'] = users
    df['message'] = other_texts

    # Display the updated DataFrame
    #print(df)

    return df