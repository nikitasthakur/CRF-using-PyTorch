import os
import gzip
import csv
import numpy as np
from pathlib import Path


class DataLoader:
    def __init__(self):
        cwd = os.getcwd()
        parent = os.path.dirname(cwd)
        data_path = parent + "\\data\\letter.data.gz"
        lines = self._read(data_path)
        self.data, self.target,self.next_ = self._parse(lines)

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target, next_ = [], [],[]
        # next_ = None

        for line in lines:
            next_.append(int(line[2]))
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data.append(pixels)
            target.append(line[1])
        return np.array(data), np.array(target),np.array(next_)

    @staticmethod
    def _pad(data, target):
        """
        Add padding to ensure word length is consistent
        """
        max_length = max(len(x) for x in target)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)

def get_dataset():
    dataset = DataLoader()
    
    # Flatten images into vectors.
    # dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))

    # One-hot encode targets.
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        # if letter:
        target[index][ord(letter) - ord('a')] = 1
    dataset.target = target

# Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    dataset.next_ = dataset.next_[order]
    return dataset


if __name__ == "__main__":
    d = get_dataset()
    
