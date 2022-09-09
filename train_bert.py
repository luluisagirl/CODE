import torch
import os
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

data_uncomplaint = pd.read_csv('data/neg_train.csv')
data_uncomplaint['label'] = 0
data_complaint = pd.read_csv('data/pos_train.csv')
data_complaint['label'] = 1
data = pd.concat([data_uncomplaint[:100], data_complaint[:100]], axis=0).reset_index(drop=True)
X = data
y = data.label.values  
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  
            add_special_tokens=True,  
            max_length=MAX_LEN,  
            padding='max_length',  
            return_attention_mask=True 
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

#tokenize
encoded_tweet = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
MAX_LEN = max([len(sent) for sent in encoded_tweet])
train_inputs, train_masks = preprocessing_for_bert(X_train)
test_inputs, test_masks = preprocessing_for_bert(X_test)

#Create PyTorch DataLoader
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)
batch_size = 4
#  DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

class BertClassifier(nn.Module):
    def __init__(self, ):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 100, 5
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), 
            nn.Softmax(dim=None),  
            nn.Linear(H, D_out) 
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

def initialize_model(epochs=2):
    bert_classifier = BertClassifier()
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=2e-5,  
                      eps=1e-8  
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

loss_fn = nn.CrossEntropyLoss()  


def train(model, train_dataloader, test_dataloader=None, epochs=2, evaluation=False):

    for epoch_i in range(epochs):
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        avg_train_loss = total_loss / len(train_dataloader)
        if evaluation:  
            test_loss, test_accuracy = evaluate(model, test_dataloader)
            time_elapsed = time.time() - t0_epoch
def evaluate(model, test_dataloader):
    model.eval()
    test_accuracy = []
    test_loss = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask) 
        loss = loss_fn(logits, b_labels.long())
        test_loss.append(loss.item())
        preds = torch.argmax(logits, dim=1).flatten()  
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)
    return val_loss, val_accuracy

bert_classifier, optimizer, scheduler = initialize_model(epochs=30)
train(bert_classifier, train_dataloader, test_dataloader, epochs=30, evaluation=True) 
net = BertClassifier()
