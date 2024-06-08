import re
import time
import numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression

dataset = load_from_disk('./datasets')
train_data = dataset['train']
test_data = dataset['test']
unsupervised_data = dataset['unsupervised']

from transformers import BertTokenizer, BertModel
model_path = "./base_uncased/"
# base-uncased
# large-uncased
# base-multilingual-uncased
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

result = []
train_number_single = 125
num1 = 0
num0 = 0
train_label = []


for i in train_data :
    if i['label'] == 1 :
        if num1 == train_number_single :
            continue
        num1 = num1 + 1
        train_label.append(1)
    else :
        if num0 == train_number_single :
            continue
        num0 = num0 + 1
        train_label.append(0)
    encoded_input = tokenizer(i['text'], padding='max_length', truncation=True, return_tensors='pt')
    output = model(**encoded_input)
    result.append(output[0].detach().numpy()[0])

print(result[0].shape)
print(result[1].shape)
print(result[2].shape)

X_train = np.array(result)
X_train_2D = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
Y_train = np.array(train_label)

log_model = LogisticRegression()
log_model.fit(X_train_2D, Y_train)

res = 0
tot = 0
X_list = []

for i in range(2*train_number_single) :
    encoded_input = tokenizer(test_data[i]['text'], padding='max_length', truncation=True, return_tensors='pt')
    output = model(**encoded_input)
    X_list.append(output[0].detach().numpy()[0])

X_test = np.array(X_list)
X_test_2D = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_test = log_model.predict(X_test_2D)

for i in range(2*train_number_single) :
    tot = tot + 1
    if Y_test[i] == test_data[i]['label'] :
        res = res + 1

print(res/tot)