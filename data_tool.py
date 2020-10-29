import os
import pickle
import torch
import csv

#处理原始数据，将最低一级的域名取出
def process_meta_data(path):
    for filename in os.listdir(path +  "/meta_dga/"):
        if not filename.endswith(".txt"):
            continue
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

#处理正常域名，数据源是Alexa前一百万活跃注册域名
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

#生成字符字典
def char_diction(path):
    #注意0是留给padding的
    if os.path.isfile("dictionary.pkl"):
        f = open("dictionary.pkl",'rb')
        char_dict = pickle.load(f)
        f.close()
    else:
         char_dict = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':10,
                 'a': 11,'b':12,'c':13,'d':14,'e':15,'f':16,'g':17,'h':18,'i':19,'j':20,'k':21,
                 'l': 22,'m':23,'n':24,'o':25,'p':26,'q':27,'r':28,'s':29,'t':30,'u':31,'v':32,'w':33,'x':34,'y':35,'z':36,
                 'A': 37, 'B': 38, 'C': 39, 'D': 40, 'E': 41, 'F': 42, 'G': 43, 'H': 44, 'I': 45, 'J': 46, 'K': 47,
                 'L': 48, 'M': 49, 'N': 50, 'O': 51, 'P': 52, 'Q': 53, 'R': 54, 'S': 55, 'T': 56, 'U': 57, 'V': 58,
                 'W': 59, 'X': 60, 'Y': 61, 'Z': 62}
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

#将域名转化为token
def tokens(path, char_dict):
    print("preparing tokens")
    f_o = open("./token_overview.txt",'w')
    token_data = []
    i = 0
    dga_path = path + "/dga"
    pkl_path = path + "/dga_tokens"
    for filename in os.listdir(dga_path):
        single_token_data = []
        if not filename.endswith(".txt"):
            continue
        f = open(dga_path + "/" + filename, 'r')
        f2 = open(pkl_path + "/" + filename + ".pkl", "wb")
        print(filename)
        lines = 0
        for line in f.readlines():
            token = []
            for char in line:
                if char in char_dict:
                    token.append(char_dict[char])
            single_token_data.append((i, torch.Tensor(token)))
            token_data.append((i, torch.Tensor(token)))
            lines = lines + 1
        f_o.write(str(i)+": " + os.path.splitext(filename)[0] + "    " + str(lines) + "\n")
        i = i + 1
        pickle.dump(single_token_data,f2)
        f2.close()
        f.close()
    f_o.close()
    return token_data


if __name__ == '__main__':
    ##处理正常域名，参数为目录和需要取的正常域名数量，注意根目录位置
    #process_normal_data("./data", 30000)
    ##如果没有处理初始域名就执行
    #process_meta_data("./data")
    ##生成字符字典
    char_dict = char_diction("./data/dga")

    ##直接用字符字典
    #f = open("dictionary.pkl", 'rb')
    #char_dict = pickle.load(f)
    #f.close()
    ##

    ##保存字典
    f = open("dictionary.pkl",'wb')
    pickle.dump(char_dict,f)
    f.close()
    token_data = tokens("./data",char_dict)
    ##保存tokens
    f = open("tokens.pkl",'wb')
    pickle.dump(token_data,f)
    f.close()
