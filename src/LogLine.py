import re
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.data import find
import logging

import sys
sys.path.append('../../../spell/pyspell')
import spell as s
import pickle

import severity_codes, facility_codes
import dangerous_directories

import PerimeterViolation
import LogKeysManager
import SentenceEmbedding

class LogLine():
    def __init__(self):
        # Classification attributes
        self.type = None # can be 'cron', 'secure', ecc
        self.raw = None # raw log line
        self.msg = None # contains the df['message']

        # Attributes
        self.severity = None
        self.facility = None
        self.time = None
        self.common_fp_length = None
        self.perimeter_violation = None
        self.temporary_log_key = None
        self.log_key = None
        self.spell_log_key = None
        self.embedding = None

        # Folders
        self.log_key_file_path = 'log_key.json'

        # Initializations
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            find('corpora/words.zip')
        except LookupError:
            nltk.download('words')
        except Exception as e:
            logging.error(f"An error occurred while checking or downloading the 'words' corpus: {e}")


    def print_attributes(self):
        attributes = [
            'type', 'raw', 'msg', 'severity', 'facility', 'time',
            'common_fp_length', 'perimeter_violation', 'temporary_log_key',
            'log_key', 'spell_log_key', 'embedding'
        ]
        
        for attr in attributes:
            print(f"{attr}: {getattr(self, attr)}")

    def get_attributes_dict(self):
        attributes = [
            'type', 'raw', 'msg', 'severity', 'facility', 'time',
            'common_fp_length', 'perimeter_violation', 'temporary_log_key',
            'log_key', 'spell_log_key', 'embedding'
        ]
        
        attr_dict = {attr: getattr(self, attr) for attr in attributes}
        return attr_dict

    def find_file_paths(self, line):
        '''
        Function that finds if there are file paths in a log line.

        Args:
            line (str): the log line

        Returns:
            file_paths (list): the list of found file paths
        '''
        file_path_pattern = r'\/(?:[^\/\s]+\/?)+\b'
        sub_pattern = r'*'
        file_paths = re.findall(file_path_pattern, line)
        return file_paths if len(file_paths) > 0 else -1


    def file_path_length(self, file_path):
        '''
        Function that calculates length of a file path.

        Args:
            line (str): the log line

        Returns:
            num_dir (int): the length of the file path
        '''
        directories = file_path.strip("/").split("/")
        num_dir = len(directories)
        return num_dir
        

    def remove_numbers_and_special_characters(self, input_string):
        # Define the regex pattern to match any character that is not a letter
        pattern = r'[^a-zA-Z\s]'
        # Replace all matches with an empty string
        cleaned_string = re.sub(pattern, ' ', input_string)
        # Define the regex pattern to match two or more spaces
        pattern = r'\s{2,}'
        # Replace all matches with a single space
        reduced_string = re.sub(pattern, ' ', cleaned_string)
        #self.temporary_log_key = reduced_string
        return reduced_string

    def remove_nonsense_words(self, log_text):
        english_words = set(words.words())
        processed_words = []
        
        for word in log_text.split():
            # Remove words that are too short or too long
            if len(word) < 3 or len(word) > 15:
                continue
            
            # Remove words with unusual character distributions
            if sum(c.isalpha() for c in word) / len(word) < 0.5:
                continue
            
            # Remove words that are not in the English dictionary
            if word.lower() not in english_words:
                continue
            
            processed_words.append(word)
        
        return ' '.join(processed_words)

    def decode_json(self, line):
        index = line.find('{')
        if index != -1:
            json_data = line[index:]
            data = json.loads(json_data)
        return data

    def nltk_word_tokenizer(self, input_string):
        # Use nltk's word_tokenize to split the string into tokens
        tokens = word_tokenize(input_string)
        return tokens

    def readline(self, line, log_type):
        self.type = log_type
        self.raw = line
        
        # extract the json parameters
        data = self.decode_json(line)
        self.severity = severity_codes.severity_dict[data['severity']]
        self.facility = facility_codes.facility_dict[data['facility']]
        self.time = data['time']
        self.msg = data['message']
        
        # find cleaned msg string
        cleaned_msg = self.preprocess_by_type()
        self.temporary_log_key = cleaned_msg

        # Build Log Key
        l = LogKeysManager.LogKeysManager(self.log_key_file_path, self.temporary_log_key)
        self.log_key = l.save_temp_log_key(self.log_key_file_path)
        
        # Build Spell Log Key
        #s.load_spell_model()
        #self.spell_log_key = s.get_log_key(self.temporary_log_key).get_id()
        
        # Build Word Embedding
        msg_tokens = self.nltk_word_tokenizer(cleaned_msg)
        s_emb = SentenceEmbedding.SentenceEmbedding()
        self.embedding = s_emb.get_bert_embedding(cleaned_msg)

        attributes_dict = self.get_attributes_dict()
        return attributes_dict


      
    def preprocess_by_type(self):
        
        line = self.msg
        if self.type == 'cron':

            try:
                pattern = r'\((.*?)\) ([a-zA-Z]+) \((.*?)\)'
                match = re.match(pattern, line)
                user_cron = match.group(1)
                cmd_cron = match.group(2)
                other_text = match.group(3)
    
                self.msg = other_text
            except:
                self.msg = line
                
            return self.preprocess()
            
        elif self.type == 'laurel':
            complex_id_pattern = r'"\d+\.\d+:\d+",'
            line = re.sub(complex_id_pattern, '{', self.msg, count=1)
            index = line.find('{')
            if index != -1:
                json_data = line[index:]
                try:
                    data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    data = -1
            return self.preprocess_laurel(data)
        else:
            return self.preprocess()

    def preprocess(self):
        # routine for non laurel preprocessing
        # Extract fp common length
        file_paths = self.find_file_paths(self.msg)
        try:
            common_prefix = os.path.commonprefix(file_paths)
            common_directory = os.path.dirname(common_prefix)
            fp_length = self.file_path_length(common_directory)
        except:
            common_prefix, common_directory, fp_length = -1, -1, -1
        self.common_fp_length = fp_length
        
        # Extract perimeter violation
        p = PerimeterViolation.PerimeterViolation()
        n, n_cron = p.find_dangerous_directories(self.msg)
        if self.type == 'cron':
            self.perimeter_violation = n
        else:
            self.perimeter_violation = (n or n_cron)
            
        # Clean the string
        if self.type == 'secure':
            cleaned_string = self.remove_nonsense_words(self.msg)
            cleaned_string = self.remove_numbers_and_special_characters(cleaned_string)
        else:
            cleaned_string = self.remove_numbers_and_special_characters(self.msg)
        return cleaned_string


    def preprocess_laurel(self, data):
        try:
            suid = parsed['SYSCALL']['suid']
        except:
            suid = -1
            
        try:
            comm = parsed['SYSCALL']['comm']
        except:
            comm = -1

        try:
            parent_comm = parsed['PARENT_INFO']['comm']
        except:
            parent_comm = -1

        try:
            exit = parsed['SYSCALL']['exit']
        except:
            exit = -1
                
        try:
            #suid = parsed['SYSCALL']['suid']
            cap_fp = parsed['PATH'][0]['cap_fp'] # extracting binary capabilities
            cap_fp = int(cap_fp, 0)
        except:
            cap_fp = -1

        perimeter = PerimeterViolation.PerimeterViolation().analyze_line(line)
        params = [suid, cap_fp, comm, parent_comm, exit]
        self.temporary_log_key = '*'
        log_key_number = -1
        return params + [log_key_number] + perimeter