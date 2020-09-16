# -*- coding: utf-8 -*-

import argparse
from collections import Counter
import code
import os
import gc
import logging
from tqdm import tqdm, trange
import random
import codecs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

import numpy as np
import pandas as pd


def set_seed(seed=4321):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calc_acc(real_label, pred_label):
    real_label = np.array(real_label)
    pred_label = np.array(pred_label)

    assert real_label.shape == pred_label.shape
    assert real_label.shape[0] % 2 == 0  #真实label应该是2的倍数

    # 对每个label：预测正确的个数/预测总数（无论真假label）
    label_acc = float((real_label == pred_label).sum()) / float(pred_label.shape[0])
    
    #
    real_label = real_label.reshape(-1,2)
    assert real_label.shape[0] == real_label[:,0].sum()

    # 转成真实问题数*（2个预测概率）
    pred_label = pred_label.reshape(-1,2)

    # 看看对每个问题预测成2个答案的概率
    #pred_idx = pred_label.argmax(dim=-1)
    pred_idx = np.argmax(pred_label, axis=1).flatten()

    # 对每个问题：预测正确的acc
    question_acc = float((pred_idx == 0).sum()) / float(pred_idx.shape[0])

    return question_acc, label_acc


class SimInputText:
    def __init__(self, id, question, attribute, label=None):
        self.id = id
        self.question = question
        self.attribute = attribute
        self.label = label


class SimInputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class preProcessor:
    def get_train_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self,data_dir):
        return self._create_examples(
            os.path.join(data_dir, "test.txt"))

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, path):
        examples = []
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if 4 == len(tokens):
                    examples.append(
                        SimInputText(id=int(tokens[0]),
                                     question=tokens[1],
                                     attribute=tokens[2],
                                     label=int(tokens[3]))
                                    )
        f.close()
        return examples


def bertEncode(texts, tokenizer, max_seq_length=512, label_list=None):

    all_tokens = []
    all_masks = []
    all_segments = []
    features = []
    
    for text in texts:
        textA = tokenizer.tokenize(text.question)
        textB = tokenizer.tokenize(text.attribute)
        idsA = tokenizer.convert_tokens_to_ids(textA)
        idsB = tokenizer.convert_tokens_to_ids(textB)
        # cls + idsA + sep + idsB + sep
        input_ids = tokenizer.build_inputs_with_special_tokens(idsA, idsB)
        masks = [1] * len(input_ids)
        #token_type_ids = tokenizer.create_token_type_ids_from_sequences(idsA, idsB)
        token_type_ids = [0] * (len(idsA) + 2) + [1] * (len(idsB) + 1)
        
        # pad seq to max length
        pad_seq = [0] * (max_seq_length - len(input_ids))
        input_ids += pad_seq
        masks += pad_seq
        token_type_ids += pad_seq

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(masks) == max_seq_length, "Error with input length {} vs {}".format(len(masks), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids), max_seq_length)

        features.append(
            SimInputFeatures(input_ids, masks, token_type_ids, text.label)
        )

    return features


def load_and_cache_example(args, tokenizer, processor, data_type):

    doc_list = ['train','dev','test']

    if data_type not in doc_list:
        raise ValueError("data_type must be one of {}".format(" ".join(doc_list)))

    cached_features_file = "cached_{}_{}".format(data_type, str(args["max_seq_length"]))
    cached_features_file = os.path.join(args["data_dir"], cached_features_file)

    # 加载feature 
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)

    else:
        label_list = processor.get_labels()

        if doc_list[0] == data_type:
            examples = processor.get_train_examples(args["data_dir"])
        elif doc_list[1] == data_type:
            examples = processor.get_dev_examples(args["data_dir"])
        elif doc_list[2] == data_type:
            examples = processor.get_test_examples(args["data_dir"])

        features = bertEncode(texts=examples, tokenizer=tokenizer, max_seq_length=args["max_seq_length"], label_list=label_list)

        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)

    return dataset


def evaluate_and_save_model(args, model, val_dataloader, epoch, best_acc):

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    val_loss, question_acc, label_acc = evaluate(args, model, val_dataloader)
    print("EPOCH : [{}/{}]   val_loss : {:.4f}   question_acc : {:.4f}   label_acc : {:.4f}"   
           .format(epoch + 1, args["epochs"], val_loss, question_acc, label_acc))
    
    if question_acc > best_acc:
        best_acc = question_acc
        # hasattr判断对象是否有某属性 bool
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained(args["output_dir"])

        #torch.save(model.state_dict(), os.path.join(args["output_dir"], "pytorch_model.bin"))

        #print("question_acc : {:.4f}".format(best_acc))

    return best_acc


def evaluate(args, model, val_dataloader):

    print("--------------- Validation ---------------")
    print("  Num examples = {}".format(len(val_dataloader)))
    print("  Batch size = {}".format(args["batch_size"]))

    total_loss = 0.0       
    total_sample = 0       # 样本数
    all_real_labels = []    
    all_pred_labels = []   

    for batch in val_dataloader:
        model.eval()
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'labels':batch[3],
            }

            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            #total_loss += loss * batch[0].shape[0]  # loss * 样本数
            total_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].to('cpu').numpy()

            total_sample += batch[0].shape[0]       # 记录样本个数

            #pred = logits.argmax(dim=-1).tolist()   # 得到预测label (list)
            pred = np.argmax(logits, axis=1).flatten()

            all_pred_labels.extend(pred)                        # 预测label
            all_real_labels.extend(batch[3].view(-1).tolist())  # 真实label

    avg_loss = total_loss / total_sample

    question_acc, label_acc = calc_acc(all_real_labels, all_pred_labels)

    return avg_loss, question_acc, label_acc


def train(args, train_dataloader, val_dataloader, model):

    # optimizer parameter
    gradient_accumulation_steps = 1
    total_lens = len(train_dataloader) // gradient_accumulation_steps * args["epochs"]

    no_decay = ['bias', 'LayerNorm.weight','transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_lens)

    print("--------------- Training ---------------")
    print("  Num Batch = {}".format(len(train_dataloader)))
    print("  Num Epochs = {}".format(args["epochs"]))
    print("  Gradient Accumulation steps = {}".format(gradient_accumulation_steps))
    print("  Total optimization steps = {}".format(total_lens))

    model.zero_grad()
    set_seed()
    train_acc = 0.0

    for i in range(int(args["epochs"])):

        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args["device"]) for t in batch)
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'labels':batch[3],
            }
            outputs = model(**inputs)

            loss, logits = outputs[0], outputs[1]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #logging_loss += loss.item()
            tr_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                #global_step += 1
                #print("EPOCH : [{}/{}]  global step : {}  loss = {:.4f}".format(i+1, args["epochs"], global_step, logging_loss))
                #logging_loss = 0.0

                # 每100步，评估一次
                #if (global_step % 50 == 0 and global_step <= 100) or (global_step % 100 == 0 and global_step <= 1000) \
                #     or (global_step % 200 == 0):
                if (step + 1) % 100 == 0:
                    print("EPOCH : [{}/{}]   step : {}   cur_loss : {:.4f} ".format(i+1, args["epochs"], step, loss.item()))
                    #train_acc = evaluate_and_save_model(args, model, val_dataloader, i, step, train_acc)

        train_acc = evaluate_and_save_model(args, model, val_dataloader, i, train_acc)
        print("                  avg_loss : {:.4f} ".format(tr_loss/len(train_dataloader)))    

    # 最后循环结束 再评估一次
    train_acc = evaluate_and_save_model(args, model, val_dataloader, i, train_acc)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_test = {"data_dir": '/content/gdrive/My Drive/nlpqa2/input/data/sim_data/',
                 "load_path": '/content/gdrive/My Drive/nlpqa2/output/data/',
                 #"vocab_file": 'bert-base-chinese-vocab.txt'
                 #"model_config": 'pytorch_model.bin',
                 #"model_path": 'config.json',
                 "max_seq_length": 128,
                 "batch_size": 16,
                 "learning_rate": 6e-6,
                 "epochs": 3,
                 "device": device
                }

    tokenizer = BertTokenizer(os.path.join(args_test["data_dir"], 'bert-base-chinese-vocab.txt'), return_tensors='pt')

    processor = preProcessor()
    test_dataset = load_and_cache_example(args_test, tokenizer, processor, 'test')

    bert_config = BertConfig.from_pretrained(os.path.join(args_test["load_path"], 'config.json'))
    bert_config.num_labels = len(processor.get_labels())

    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args_test["batch_size"])

    model_kwargs = {'config': bert_config, "from_tf": False}
    model = BertForSequenceClassification.from_pretrained(os.path.join(args_test["load_path"], 'pytorch_model.bin'), **model_kwargs)
    #model.load_state_dict(torch.load(os.path.join(args["load_path"], "model_path")))

    model = model.to(device)

    del test_dataset 
    gc.collect()

    total_loss = 0.0      
    total_sample = 0  # 样本数
    all_real_labels = []    
    all_pred_labels = []   

    for batch in test_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]
                     }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            total_loss += loss.item()
            #total_loss += loss * batch[0].shape[0]  # loss * 样本数
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].to('cpu').numpy()

            total_sample += batch[0].shape[0]  # 记录样本个数

            #pred = logits.argmax(dim=-1).tolist()  # 得到预测的label转为list
            pred = np.argmax(logits, axis=1).flatten()

            all_pred_labels.extend(pred) 
            all_real_labels.extend(batch[3].view(-1).tolist())  

    loss = total_loss / total_sample
    question_acc, label_acc = calc_acc(all_real_labels, all_pred_labels)

    print("avg_loss", loss)
    print("question_acc", question_acc)
    print("label_acc", label_acc)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device", device)

    args_train = {"data_dir": '/content/gdrive/My Drive/nlpqa2/input/data/sim_data/',
            "pre_train_model": 'bert-base-chinese',
            "output_dir": '/content/gdrive/My Drive/nlpqa2/output/data/',
            "max_seq_length": 128,
            "batch_size": 16,
            "learning_rate": 6e-6,
            "epochs": 3,
            "device": device
    }

    assert os.path.exists(args_train["data_dir"])

    processor = preProcessor()

    tokenizer = BertTokenizer.from_pretrained(args_train["pre_train_model"], return_tensors='pt')  

    train_dataset = load_and_cache_example(args_train, tokenizer, processor, 'train')
    val_dataset = load_and_cache_example(args_train, tokenizer, processor, 'dev')
    
    bert_config = BertConfig.from_pretrained(args_train["pre_train_model"])
    bert_config.num_labels = len(processor.get_labels())

    model_kwargs = {'config': bert_config, "from_tf": True}
    model = BertForSequenceClassification.from_pretrained(args_train["pre_train_model"], **model_kwargs)
    
    model = model.to(device)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args_train["batch_size"])
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args_train["batch_size"])
    
    del train_dataset, val_dataset 
    gc.collect()

    train(args_train, train_dataloader, val_dataloader, model)
    
    del train_dataloader, val_dataloader
    gc.collect()
    #test()