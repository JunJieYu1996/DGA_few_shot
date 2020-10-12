import pickle
import random
import string

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd

def collate_fn_2(batch):
    lens = [i[1].shape[0] for i in batch]
    bsz, max_len = len(batch),  max(lens)

    data_tensor = torch.zeros(bsz, max_len, dtype=torch.long)
    label_tensor = torch.Tensor([i[0] for i in batch])

    for b_ix, b in enumerate(batch):
        data_tensor[b_ix, :b[1].shape[0]] = b[1].unsqueeze(0)

    return data_tensor, label_tensor

def split_train_test():
    door_number = 100
    split_number = int(door_number * 0.8)
    class_limit = 10
    # names = ["banjori","emotet","pykspa","rovnix","tinba"]
    # '''
    names = []
    i = 0
    f = open("token_overview.txt", "r")
    for line in f.readlines():
        elements = line.split()
        if int(elements[2]) > door_number:
            names.append(elements[1])
            i = i + 1
        if i == class_limit:
            break
    f.close()
    # '''

    train_data = []
    test_data = []
    i = 0
    for name in names:
        full_name = "./data/dga_tokens/" + name + ".txt.pkl"
        f = open(full_name, "rb")
        token = pickle.load(f)
        change_token = []
        for _ in token:
            change_token.append((i, _[1]))
        random.shuffle(change_token)
        new_token = change_token[:door_number]
        train_data.extend(new_token[:split_number])
        test_data.extend(new_token[split_number:])
        i = i + 1
        f.close()
    print(names)
    print("door number:" + str(door_number))
    print("split number:" + str(split_number))
    print("---------------------------------")
    print("class num: " + str(i))
    print("total num: " + str(i * door_number))
    print("train_num: " + str(i * split_number))
    print("test_num: " + str(i * (door_number-split_number)))

    return train_data, test_data, len(names)

def split_train_test_2():
    door_number = 100
    #split_number = int(door_number * 0.8)
    class_limit = 30
    support_class_num = 20
    # '''
    names = []
    i = 0
    f = open("token_overview.txt", "r")
    for line in f.readlines():
        elements = line.split()
        if int(elements[2]) > door_number:
            names.append(elements[1])
            i = i + 1
        if i == class_limit:
            break
    f.close()
    # '''
    random.shuffle(names)

    '''
    ['qadars']
    ['ramnit', 'banjori', 'pykspa', 'gameover', 'locky', 'necurs', 'normal', 'ranbyus', 'emotet', 'murofet']
    '''
    #special
    #support_class = ['qadars']
    #val_class = ['ramnit', 'banjori', 'pykspa', 'gameover', 'locky', 'necurs', 'normal', 'ranbyus', 'emotet', 'murofet']
    support_class = names[:support_class_num]
    val_class = names[support_class_num:]
    train_data = []
    test_data = []
    i = 0
    for single_class in support_class:
        full_name = "./data/dga_tokens/" + single_class+ ".txt.pkl"
        f = open(full_name, "rb")
        token = pickle.load(f)
        change_token = []
        for _ in token:
            change_token.append((i, _[1]))
        random.shuffle(change_token)
        new_token = change_token[:door_number]
        train_data.extend(new_token)
        i = i + 1
        f.close()

    i = 0
    for single_class in val_class:
        full_name = "./data/dga_tokens/" + single_class+ ".txt.pkl"
        f = open(full_name, "rb")
        token = pickle.load(f)
        change_token = []
        for _ in token:
            change_token.append((i, _[1]))
        random.shuffle(change_token)
        new_token = change_token[:door_number]
        test_data.extend(new_token)
        i = i + 1
        f.close()

    print(support_class)
    print(val_class)
    print("door number:" + str(door_number))
    print("class limit:" + str(support_class_num))
    print("---------------------------------")
    print("total num: " + str(class_limit * door_number))
    print("support_num: " + str(support_class_num * door_number))
    print("val_num: " + str((class_limit-support_class_num) * door_number))

    return train_data, test_data, (class_limit-support_class_num)

def collate_fn(batch):
    left_lens = [i['left'].shape[0] for i in batch]
    right_lens = [i['right'].shape[0] for i in batch]
    bsz, max_right_len, max_left_len = len(batch),  max(right_lens), max(left_lens)

    # #22是手算cnn的最小输入长度，不然会报错
    # if max_left_len < 22:
    #     max_left_len = 22
    # if max_right_len < 22:
    #     max_right_len = 22
    left_tensor = torch.zeros(bsz, max_left_len, dtype=torch.long)
    right_tensor = torch.zeros(bsz, max_right_len, dtype=torch.long)
    label_tensor = torch.Tensor([i['label'] for i in batch])

    for b_ix, b in enumerate(batch):
        left_tensor[b_ix, :b['left'].shape[0]] = b['left'].unsqueeze(0)
        right_tensor[b_ix, :b['right'].shape[0]] = b['right'].unsqueeze(0)

    return left_tensor, right_tensor, label_tensor

class Pair_Loader(Dataset):
    def __init__(self, data, base_num):
        self.labs_to_sample = []
        self.data = data
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
        pos_examples = []
        neg_examples = []
        class_num = len(self.lab_to_data.items())
        data_per_class = int(self.base_num/class_num)
        for class_index in range(class_num):
            base_data = random.sample(self.lab_to_data[class_index], data_per_class)
            print("matching positive samples")
            sample_data = [random.choice(self.lab_to_data[class_index])for _ in range(data_per_class*(class_num-1))]
            #sample_data = random.sample(self.lab_to_data[class_index],data_per_class*2)
            for i in range(data_per_class):
                for j in range(data_per_class*(class_num-1)):
                    example_dict ={"left": base_data[i],
                                   "right": sample_data[j],
                                   "label": 1}
                    pos_examples.append(example_dict)
            print("matching negative samples")
            for k,v in self.lab_to_data.items():
                if k != class_index:
                    sample_data = random.sample(self.lab_to_data[k],data_per_class)
                    for i in range(data_per_class):
                        for j in range(data_per_class):
                            example_dict ={"left": base_data[i],
                                           "right": sample_data[j],
                                           "label": 0}
                            neg_examples.append(example_dict)
        pos_examples = random.sample(pos_examples, int(len(pos_examples)/200))
        neg_examples = random.sample(neg_examples, int(len(neg_examples)/200))
        examples.extend(pos_examples)
        examples.extend(neg_examples)
        print("total:"+ str(len(examples)))
        return examples

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        return self.pair_data[index]


if __name__ == '__main__':
    split_train_test_2()