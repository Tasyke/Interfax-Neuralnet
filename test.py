import json
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils



def read_data(data):
    text= []
    arr = []
    title = []
    for news in range(0,150):
        title.append(data[news]['title'])
        for headline in data[news]['news']:
            if headline['headline'] is None:
                arr.append(headline['body'])
            else:
                arr.append(headline['headline'])
                arr.append(headline['slugline'])
                arr.append(headline['body'])
        text.append(arr)
        arr = []
    return (text,title)

with open('dataset_public.json', 'r', encoding='UTF-8') as file:
    data = json.load(file)

raw_data = read_data(data)
text = raw_data[0]
title = raw_data[1]

print(text[0])
# with open('data.txt','w',encoding='UTF-8') as file:
#     for i in text:
#         for j in i:
#             file.write(j)
#             file.write('\n')

# filename = "data.txt"
# raw_text = open(filename,'r',encoding='utf-8').read()
# raw_text = raw_text.lower()

lst_no = ['.', ',', ':', '!', '"', "'", '[', ']', '-', '—', '(', ')', '?', '_', '`'  ]   # и т.д.
lst = []


# for word in text.lower().split():
#     if not word in lst_no:
#         _word = word 
#         if word[-1] in lst_no:
#             _word = _word[:-1]
#         if word[0] in lst_no:
#             _word = _word[1:] 
#         lst.append(_word)

# _dict = dict()
# for word in lst:
#     _dict[word] = _dict.get(word, 0) + 1

# # сортируем словарь посредством формирования списка (значение, ключ)
# _list = []
# for key, value in _dict.items():
#     _list.append((value, key))
#     _list.sort(reverse=True)

# # самое частое слово в этом тексте
# print(f'самое частое слово в этом тексте -> `{_list[0][1]}`, использовалось `{_list[0][0]}` раз.')