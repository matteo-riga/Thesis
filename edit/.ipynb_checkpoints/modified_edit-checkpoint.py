#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import pickle

class EditDistanceObj:

    def __init__(self, objid, seq, lineid, refmt):
        self._refmt = refmt
        if isinstance(seq, str):
            self._sequence = re.split(self._refmt, seq.strip())
        else:
            self._sequence = seq
        self._lineids = [lineid]
        self._id = objid

    def is_number_or_special(self, char):
        return char.isdigit() or not char.isalnum()

    def char_distance(self, char1, char2):
        if self.is_number_or_special(char1) and self.is_number_or_special(char2):
            return 0
        return 0 if char1 == char2 else 1

    def edit_distance(self, seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = self.char_distance(seq1[i-1], seq2[j-1])
                dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                               dp[i][j-1] + 1,      # insertion
                               dp[i-1][j-1] + cost) # substitution
        
        return dp[m][n]

    def get_distance(self, seq):
        if isinstance(seq, str):
            seq = re.split(self._refmt, seq.strip())
        return self.edit_distance(self._sequence, seq)

    def insert(self, seq, lineid):
        if isinstance(seq, str):
            seq = re.split(self._refmt, seq.strip())
        self._lineids.append(lineid)
        # We don't modify the sequence for edit distance

    def tojson(self):
        ret = {
            "sequence": " ".join(self._sequence),
            "lineids": self._lineids
        }
        return json.dumps(ret)

    def length(self):
        return len(self._sequence)

    def get_id(self):
        return self._id

class EditDistanceMap:

    def __init__(self, refmt):
        self._refmt = refmt
        self._objs = []
        self._lineid = 0
        self._id = 0

    def insert(self, entry):
        seq = re.split(self._refmt, entry.strip())
        obj = self.match(seq)
        if obj is None:
            self._lineid += 1
            obj = EditDistanceObj(self._id, seq, self._lineid, self._refmt)
            self._objs.append(obj)
            self._id += 1
        else:
            self._lineid += 1
            obj.insert(seq, self._lineid)

        return obj

    def match(self, seq):
        if isinstance(seq, str):
            seq = re.split(self._refmt, seq.strip())
        best_match = None
        best_distance = float('inf')
        seq_len = len(seq)
        
        for obj in self._objs:
            obj_len = obj.length()
            if abs(obj_len - seq_len) > min(obj_len, seq_len) / 2:
                continue

            distance = obj.get_distance(seq)
            if distance < best_distance:
                best_match = obj
                best_distance = distance
        
        return best_match

    def objat(self, idx):
        return self._objs[idx]

    def size(self):
        return len(self._objs)

    def dump(self):
        for i, obj in enumerate(self._objs):
            print(i, obj.tojson())

def save(filename, edit_distance_map):
    if isinstance(edit_distance_map, EditDistanceMap):
        with open(filename, 'wb') as f:
            pickle.dump(edit_distance_map, f)
    else:
        if __debug__:
            print(f"{filename} isn't an EditDistanceMap object")

def load(filename):
    with open(filename, 'rb') as f:
        edm = pickle.load(f)
        if isinstance(edm, EditDistanceMap):
            return edm
        else:
            if __debug__:
                print(f"{filename} isn't an EditDistanceMap object")
            return None