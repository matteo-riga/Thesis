import re
import json
from collections import defaultdict

class Spell:
    def __init__(self, split_rule=r'[\s:=,]+', label='*', similarity=0.5):
        self.split_rule = split_rule
        self.label = label
        self.similarity = similarity
        self.templates = []
        self.file_path = 'spell_log_key.txt'

    def write_list_to_file(self):
        # Convert the list to a dictionary with integer keys and string values
        template_dict = {i: str(item) for i, item in enumerate(self.templates, start=1)}
        with open(self.file_path, 'w') as file:
            for key, value in template_dict.items():
                file.write(f"{key}: {value}\n")

    def read_list_from_file(self):
        template_dict = {}
        with open(self.file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(": ", 1)
                template_dict[int(key)] = value
        self.templates = template_dict
        return self.templates

    def load_templates_from_file(self):
        self.templates = self.read_list_from_file()
        #print(self.templates)
        return self.templates

    def save_template_to_file(self):
        with open(self.file_path, 'w') as json_file:
            json.dump(self.templates, json_file, indent=4)

    def lcs(self, seq1, seq2):
        """Compute the longest common subsequence between two sequences."""
        lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
        for i, x in enumerate(seq1):
            for j, y in enumerate(seq2):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
        result = []
        x, y = len(seq1), len(seq2)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                y -= 1
            else:
                result.insert(0, seq1[x - 1])
                x -= 1
                y -= 1
        return result

    def tokenize(self, log):
        """Tokenize a log line based on the split rule."""
        return re.split(self.split_rule, log)

    def extract_template(self, log):
        """Extract a template from a log line."""
        log_tokens = self.tokenize(log)
        for template in self.templates:
            lcs_result = self.lcs(template, log_tokens)
            lcs_length = len(lcs_result)
            similarity = lcs_length / min(len(template), len(log_tokens))
            if similarity >= self.similarity:
                new_template = [self.label if token not in lcs_result else token for token in log_tokens]
                return new_template
        return log_tokens

    def update_templates(self, log):
        """Update the list of templates with a new log line."""
        template = self.extract_template(log)
        if template not in self.templates:
            self.templates.append(template)

    def parse_logs(self, logs):
        """Parse multiple log lines."""
        for log in logs:
            self.update_templates(log)

    def display_templates(self):
        """Display the extracted templates."""
        for template in self.templates:
            print('Template:', ' '.join(template))