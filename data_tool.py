import os
import pickle
import torch
import csv

def process_meta_data(path):
    for filename in os.listdir(path +  "/meta_dga/"):
        print(filename)
        f = open(path + "/meta_dga/" + filename, 'r')
        f2 = open(path + "/dga/" + filename, 'w')
        for line in f.readlines():
            #print(line)
            if line[0] != "\n" and line[0] != "#":
                full_domain = line.split()[0]
                domain = full_domain.split('.')[0] + "\n"
                #print(domain)
                f2.write(domain)
        f.close()
        f2.close()

def process_normal_data(path,normal_num):
    f = open(path + "/meta_dga/top-1m.csv", 'r')
    f2 = open(path + "/meta_dga/normal.txt", 'w')
    reader = csv.reader(f)
    i = 0
    for row in reader:
        f2.write(row[1]+"\n")
        i = i + 1
        if i == normal_num:
            break


def char_diction(path):
    if os.path.isfile("dictionary.pkl"):
        f = open("dictionary.pkl",'rb')
        char_dict = pickle.load(f)
        f.close()
    else:
         char_dict = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0,
                 'a': 10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,
                 'l': 21,'m':22,'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,'u':30,'v':31,'w':32,'x':33,'y':34,'z':35,
                 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46,
                 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57,
                 'W': 58, 'X': 59, 'Y': 60, 'Z': 61}
    i = len(char_dict)
    print("checking dictionary...")
    for filename in os.listdir(path):
        f = open(path+"/"+filename,'r')
        print(filename)
        for line in f.readlines():
            for char in line:
                if char not in char_dict and char != "\n":
                    print("...new char: "+ char )
                    char_dict[char] = i
                    i = i + 1
        f.close()
    return char_dict

def tokens(path, char_dict):
    print("preparing tokens")
    f_o = open("./token_overview.txt",'w')
    token_data = []
    i = 0
    for filename in os.listdir(path):
        if not filename.endswith(".txt"):
            continue
        f = open(path+"/"+filename,'r')
        print(filename)
        lines = 0
        for line in f.readlines():
            token = []
            for char in line:
                if char in char_dict:
                    token.append(char_dict[char])
            token_data.append((i,torch.Tensor(token)))
            lines = lines + 1
        f_o.write(str(i)+": " + os.path.splitext(filename)[0] + "    " + str(lines))
        i = i + 1
        f.close()
    f_o.close()
    return token_data


if __name__ == '__main__':
    ##如果没有处理初始域名就执行
    #process_meta_data("./data")
    ##生成字符字典
    #dict = char_diction("./data/dga")
    ##直接用
    f = open("dictionary.pkl", 'rb')
    char_dict = pickle.load(f)
    f.close()
    ##保存字典
    #f = open("dictionary.pkl",'wb')
    #pickle.dump(dict,f)
    #f.close()
    token_data = tokens("./data/dga",char_dict)
    ##保存tokens
    f = open("tokens.pkl",'wb')
    pickle.dump(token_data,f)
    f.close()
