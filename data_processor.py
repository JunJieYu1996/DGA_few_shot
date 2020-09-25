import pickle
import random
import string

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd

def split_train_test():
    f = open("tokens.pkl", 'rb')
    tokens = pickle.load(f)
    f.close()
    split_num = 1200
    example_per_class = 100
    new_tokens = []
    i = 0
    j = 0
    for token in tokens:
        if token[0] == i and j < example_per_class:
            new_tokens.append(token)
            j = j + 1
        elif j == 100:
            i = i + 1
            j = 0
    print("class num: " + str(i))
    print("total num: " + str(i * example_per_class))
    print("split_num: " + str(split_num))
    random.shuffle(new_tokens)
    train_data = new_tokens[:split_num]
    test_data = new_tokens[split_num:]
    return train_data, test_data

def collate_fn(batch):
    left_lens = [i['left'].shape[0] for i in batch]
    right_lens = [i['right'].shape[0] for i in batch]
    bsz, max_right_len, max_left_len = len(batch),  max(right_lens), max(left_lens)

    if max_left_len < 22:
        max_left_len = 22
    if max_right_len < 22:
        max_right_len = 22
    left_tensor = torch.zeros(bsz, max_left_len, dtype=torch.long)
    right_tensor = torch.zeros(bsz, max_right_len, dtype=torch.long)
    label_tensor = torch.Tensor([i['label'] for i in batch])

    for b_ix, b in enumerate(batch):
        left_tensor[b_ix, :b['left'].shape[0]] = b['left'].unsqueeze(0)
        right_tensor[b_ix, :b['right'].shape[0]] = b['right'].unsqueeze(0)

    return left_tensor, right_tensor, label_tensor

class Pair_Loader(Dataset):
    def __init__(self, data, base_num, few_shot:bool=True):
        self.labs_to_sample = []
        self.data = data
        self.few_shot = few_shot
        self.lab_to_data = None
        self._get_lab_data_maps()
        self.base_num = base_num
        self.pair_data = self._build_pairs()


    def get_label_freq(self):
        print(pd.Series([i['label'] for i in self.pair_data]).value_counts())

    def _get_lab_data_maps(self):
        lab_data_map = {}
        for d in self.data:
            k_d, v_d = d
            if not(k_d in lab_data_map.keys()):
                lab_data_map[k_d] = []
            lab_data_map[k_d].append(v_d)

        self.lab_to_data = lab_data_map

    def _build_pairs(self):
        examples = []
        class_num = len(self.lab_to_data.items())
        data_per_class = int(self.base_num/100)
        for class_index in range(class_num):
            base_data = random.sample(self.lab_to_data[class_index], data_per_class)
            print("matching positive samples")

            sample_data = [random.choice(self.lab_to_data[class_index])for _ in range(data_per_class*10)]
            #sample_data = random.sample(self.lab_to_data[class_index],data_per_class*2)
            for i in range(data_per_class):
                for j in range(10):
                    example_dict ={"left": base_data[i],
                                   "right": sample_data[j],
                                   "label": 1}
                    examples.append(example_dict)
            print("matching negative samples")
            for k,v in self.lab_to_data.items():
                if k != class_index:
                    sample_data = random.sample(self.lab_to_data[k],data_per_class)
                for i in range(data_per_class):
                    example_dict ={"left": base_data[i],
                                   "right": sample_data[i],
                                   "label": 0}
                    examples.append(example_dict)
        return examples

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        return self.pair_data[index]


if __name__ == '__main__':
    split_train_test()