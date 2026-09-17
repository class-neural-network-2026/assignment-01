import csv
import os

class MyResult:
    def __init__(self):
        self.key = []
        self.val = {} 

    def add_result(self, key, val):
        self.key.append(key)
        self.val[key] = val

    def __len__(self):
        return len(self.key)

    def empty(self):
        self.key = []
        self.val = {}

    def add_result(self, key, val):
        if key not in self.key:
            self.key.append(key)
        self.val[key] = val

    def get_result(self, key):
        if key in self.val:
            return self.val[key]
        else:
            return None
        
    def save(self, filename='result.csv'):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            for i in range(len(self.key)):
                key = self.key[i]
                val = self.val[key]
                wr.writerow([key, val])
           
    def load(self, filename='result.csv'):
        self.empty()
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row) == 2:
                    key = row[0]
                    val = float(row[1])
                    self.add_result(key, val)