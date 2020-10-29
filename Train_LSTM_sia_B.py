from models import *

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import random
import pickle
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from torch.utils.data import DataLoader, RandomSampler

from sklearn import metrics

####
#2020.10.28
####

def split_train_test():
    door_number = 100
    split_number = int(door_number * 0.8)
    i = 0
    train_data = []
    test_data = []
    dga_name_list = []

    for filename in os.listdir("./data/dga_tokens/"):
        print(filename)
        full_name = "./data/dga_tokens/" + filename
        f = open(full_name, "rb")
        token = pickle.load(f)
        if len(token) < door_number:
            continue
        if filename == "normal.txt.pkl":
            continue
        change_token = []
        for _ in token:
            change_token.append((0, _[1]))
        random.shuffle(change_token)
        train_data.extend(change_token[:split_number])
        test_data.extend(change_token[split_number:door_number])
        i = i + 1
        dga_name_list.append(filename.split('.')[0])
        f.close()
    print("dga_data prepared")

    f = open("./data/dga_tokens/normal.txt.pkl", "rb")
    token = pickle.load(f)
    change_token = []
    for _ in token:
        change_token.append((1, _[1]))
    random.shuffle(change_token)
    train_data.extend(change_token[:split_number * i])
    test_data.extend(change_token[split_number * i:door_number * i])

    print("normal_deta prepared")
    print(dga_name_list)

    return train_data, test_data, i

def collate_fn(batch):
    left_lens = [i['left'].shape[0] for i in batch]
    right_lens = [i['right'].shape[0] for i in batch]
    bsz, max_right_len, max_left_len = len(batch),  50, 50

    left_tensor = torch.zeros(bsz, max_right_len, dtype=torch.long)
    right_tensor = torch.zeros(bsz, max_left_len, dtype=torch.long)
    label_tensor = torch.Tensor([i['label'] for i in batch])

    for b_ix, b in enumerate(batch):
        left_tensor[b_ix, :b['left'].shape[0]] = b['left'].unsqueeze(0)
        right_tensor[b_ix, :b['right'].shape[0]] = b['right'].unsqueeze(0)

    return left_tensor, right_tensor, label_tensor, left_lens, right_lens

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
            #sample_data = [random.choice(self.lab_to_data[class_index]) for _ inrange(100)]
            for i in range(data_per_class):
                for j in range(len(sample_data)):
                    example_dict ={"left": base_data[i],
                                   "right": sample_data[j],
                                   "label": 1}
                    pos_examples.append(example_dict)
            print("matching negative samples")
            for k,v in self.lab_to_data.items():
                if k != class_index:
                    sample_data = random.sample(self.lab_to_data[k],data_per_class)
                    #sample_data = random.sample(self.lab_to_data[k], 100)
                    for i in range(data_per_class):
                        for j in range(len(sample_data)):
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

class SiameseNet_LSTM(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = 200
        self.hidden_dim = 256
        self.hidden_layers = 3
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_dropout = nn.Dropout(p=0.5)
        self.dense_dropout = nn.Dropout(p=0.5)
        self.LSTM = nn.LSTM(self.embed_dim,self.hidden_dim)
        self.pool = nn.AvgPool1d(1)
        self.dense_hidden = nn.Linear(in_features=256, out_features=512)
        self.dense_out = nn.Linear(in_features=512, out_features=1)


    def forward(self, x_left, x_right, left_lens, right_lens):
        x_left_embed = self.embed_dropout(self.embed(x_left)).permute(1, 0, 2)
        x_right_embed = self.embed_dropout(self.embed(x_right)).permute(1, 0, 2)
        left_embed = pack_padded_sequence(x_left_embed, left_lens, enforce_sorted=False)
        right_embed = pack_padded_sequence(x_right_embed, right_lens, enforce_sorted=False)

        # Run encoder
        _, (x_left, _ ) = self.LSTM(left_embed)
        _, (x_right, _)  = self.LSTM(right_embed)

        # Pooling
        minus = x_left - x_right
        final_encoded = torch.abs(minus)

        hidden_out = torch.relu(self.dense_hidden(final_encoded))
        logits_out = self.dense_out(self.dense_dropout(hidden_out))

        return logits_out

class Trainer:
    def __init__(self,train_data, test_data,char_size):
        print("initing trainer...")
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = 3
        self.char_size = char_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        self.model = SiameseNet_LSTM(vocab_size=self.char_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def _setup_data(self):
        train_set = Pair_Loader(self.train_data, base_num=len(self.train_data))
        test_set = Pair_Loader(self.test_data,base_num=len(self.test_data))
        print('Train label distribution:')
        train_set.get_label_freq()
        print('Test label distribution:')
        test_set.get_label_freq()
        train_sampler = RandomSampler(train_set)
        test_sampler = RandomSampler(test_set)
        self.train_loader = DataLoader(train_set, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_set, sampler=test_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)
    def evaluate(self, data):
        self.model.eval()

        data_len = len(data)
        total_loss = 0.0
        y_true, y_pred = [], []

        for ix, (x_left, x_right, y, left_lens, right_lens) in enumerate(data):
            # print('*', flush=True, end='')
            sys.stdout.write('\rEvaluating {} examples...'.format((ix + 1) * x_left.shape[0]))
            with torch.no_grad():
                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)

                output = self.model(x_left, x_right, left_lens, right_lens).squeeze()
                # print('output:', output.shape)
                losses = self.criterion(output, y)

                total_loss += losses.item() # .data[0]
                probs = torch.sigmoid(output).data.cpu().numpy().tolist()
                # print(type(probs))
                preds = [1 if p >= 0.5 else 0 for p in probs]
                y_pred.extend(preds)
                # print(y.data.cpu().numpy().tolist()[:10])
                y_true.extend([int(i) for i in y.data.cpu().numpy().tolist()])

        update_loss = total_loss / data_len
        print('Test loss:', update_loss)
        print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))
        print()

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(500):
            total_loss = 0.0
            print("for epoch..." + str(epoch))
            self.model.train()
            for x_left, x_right, y_batch, left_lens, right_lens in self.train_loader:
                #print('=', flush=True, end='')
                x_left, x_right, y_batch = x_left.to(self.device), x_right.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_left, x_right, left_lens, right_lens).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader)
            print()

            if test_acc > best_acc:
                print('Saving model...')
                best_acc = test_acc
                print(best_acc)
                #torch.save(self.model.state_dict(), os.path.join(self.model_path, 'conv_one_shot_model.pt'))

def main():
    f = open("dictionary.pkl", 'rb')
    char_dict = pickle.load(f)
    f.close()
    char_size = len(char_dict.items())
    train_data, test_data, _ = split_train_test()
    trainer = Trainer(train_data, test_data, char_size)
    trainer.train()

if __name__ == '__main__':
    main()