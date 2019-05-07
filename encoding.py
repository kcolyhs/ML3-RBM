'''
Module to contain the encoder class and utility functions
'''
import pickle
import csv
import numpy as np


class Encoder:
    '''
    Class that performs the encoding and the decoding for the ML3 dataset
    '''
    root = "./"
    filename = "encoder"

    def __init__(self, cutoff):
        self.cutoff = cutoff
        self.m = 0
        self.index_to_ans = []
        self.ans_to_index = []
        self.ans_widths = []

    def load_csv(self, path='ML3AllSites.csv', encoding="ISO-8859-1"):
        with open(path, 'r', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            data = list(reader)

            self.generate_mapping(data[1:])
        return data

    @staticmethod
    def load():
        '''
        Returns the encoder object saved in the file
        '''

        file = open(Encoder.root+Encoder.filename, "rb")

        encoder = pickle.load(file)
        file.close()
        return encoder

    def save(self):
        '''
        Saves self to file
        '''

        file = open(Encoder.root+Encoder.filename, "wb")

        pickle.dump(self, file)
        file.close()
        return file

    def generate_mapping(self, data):
        '''
        Creates the mapping to be used in the encoding
        Sets:
            index_to_ans = array containing the answers for each index of the
            encoding
            ans_widths = array containing how many indexes every question
            occupies
            ans_to_index = list of dictionaries mapping answer:index in encode
        '''
        self.m = len(data)
        ans_sets = [set() for _ in range(len(data[1]))]
        count_dicts = [dict() for _ in range(len(data[1]))]
        for row in data:
            for i, col in enumerate(row):
                if col == 'NA':
                    continue
                ans_sets[i].add(col)
                if col not in count_dicts[i].keys():
                    count_dicts[i][col] = 1
                else:
                    count_dicts[i][col] += 1

        index_to_ans = []
        ans_widths = np.zeros_like(data[1], np.int32)
        ans_to_index = [{} for _ in range(len(data[1]))]
        for i, ans_set in enumerate(ans_sets):
            for ans in ans_set:
                if count_dicts[i][ans] >= self.cutoff:
                    index_to_ans.append(ans)
                    ans_to_index[i][ans] = ans_widths[i]
                    ans_widths[i] += 1
        self.index_to_ans = index_to_ans
        self.ans_widths = ans_widths
        self.ans_to_index = ans_to_index

        return index_to_ans, ans_widths, ans_to_index

    def encode(self, data):
        pruned_data = np.zeros(shape=(self.m, len(ENCODER.index_to_ans)))
        existence = np.zeros_like(pruned_data)

        for i, row in enumerate(data):
            mapped_index = 0
            for j, ans in enumerate(row):
                if(ans != 'NA' and ans in self.ans_to_index[j]):
                    existence[i, mapped_index:mapped_index+self.ans_widths[j]] = True
                    pruned_data[i, mapped_index + self.ans_to_index[j][ans]] = 1
                mapped_index += self.ans_widths[j]
        return existence, pruned_data

    def decode(self, data):
        answers = []
        for row in data:
            reconstructed_ans = []
            for k in range(len(self.ans_widths)):
                start_index = np.sum(self.ans_widths[:k])
                end_index = start_index+self.ans_widths[k]
                if start_index == end_index:
                    reconstructed_ans.append("NA")
                else:
                    ans_oh = row[start_index:end_index]
                    ans_index = np.argmax(ans_oh)
                    reconstructed_ans.append(
                        self.index_to_ans[start_index+ans_index])
            answers.append(reconstructed_ans)
        return np.array(answers)

    def kth_pruned_feature(self, data, k):
        start_index = np.sum(self.ans_widths[:k])
        end_index = start_index + self.ans_widths[k]
        return data[:, start_index:end_index]


if __name__ == "__main__":
    ENCODER = Encoder(3)
    ENCODER.load_csv()
    ENCODER.save()
    ENCODER = Encoder.load()
