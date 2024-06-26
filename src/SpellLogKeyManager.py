import pandas as pd
import re
import sys

sys.path.append('../../../spell/pyspell')
import spell as s
import pickle

class SpellLogKeyManager():

    def __init__(self):
        #self.df = df
        self.spell_model = s.lcsmap('[\\s]+')


    def load_spell_model(self):
        with open('../spell/slm.pickle', 'rb') as file:
            self.spell_model = pickle.load(file)
            

    def get_message_line(self, line):
        
        # Understand if we are inside laurel
        complex_id_pattern = r'"\d+\.\d+:\d+",'
        matches = re.findall(complex_id_pattern, line)
        res = len(matches)
        # If we are inside laurel
        if res > 0 or '\"NODE\"' in line:
            return None
        # If we are inside cron
        elif 'CMD' in line:
            pattern = r'\((.*?)\) ([a-zA-Z]+) \((.*?)\)'
            match = re.match(pattern, line)

            user_cron = match.group(1)
            cmd_cron = match.group(2)
            other_text = match.group(3)
            line = other_text
        else:
            line = line

        return line


    def get_messages(self, df):
        
        lines = []
        for line in df['message']:
            l = self.get_message_line(line)
            lines.append(l)

        df = pd.DataFrame(lines)
        df.columns = ['message']

        return df


    def train_spell(self, df):
        self.df = df
        training_lines_df = self.get_messages(df)
        training_lines = training_lines_df['message']
        slm = self.spell_model
        for l in training_lines:
            sub = l.strip('\n')
            obj = slm.insert(sub)

        # Automatically save spell model
        s.save('slm.pickle', slm)
        self.spell_model = slm


    def dump_spell(self):
        self.spell_model.dump()

    def get_log_key(self, test_line):
        sub = test_line.strip('\n')
        obj = self.spell_model.match(sub)
        #print(obj.get_id(), obj.param(sub))
        return obj

    def save_model(self):
        s.save('slm.pickle', self.spell_model)