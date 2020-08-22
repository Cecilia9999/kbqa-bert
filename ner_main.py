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
import bertcrf

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd

CRF_LABELS = ["O", "B-LOC", "I-LOC"]

def calc_real_sentences(label_ids, mask, pred):
    label_ids = np.array(label_ids)
    mask = np.array(mask)
    pred = np.array(pred)
    #print('label shape: ', label_ids.shape)
    #print('mask shape: ', mask.shape)
    #print('pred shape: ', pred.shape)

    # shape (batch_size, max_len)
    assert label_ids.shape == mask.shape
    # batch_size
    assert label_ids.shape[0] == pred.shape[0]

    # 第0位是[CLS] 最后一位是<pad> 或者 [SEP]
    new_ids = label_ids[:, 1:-1]
    new_mask = mask[:, 2:]  # 保持长度和new_ids一致即可

    real_ids = []
    for i in range(new_ids.shape[0]):
        # real label的长度等于mask去头/尾/pad的长度
        seq_len = new_mask[i].sum()  
        assert seq_len == len(pred[i])
        real_ids.append(new_ids[i][:seq_len].tolist())

    return real_ids


def flatten(inputs):
    result = []
    for i in inputs:
        result.extend(i) 
    return result 


def set_seed(seed=4321):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class NerInputText:
    def __init__(self, id, question, label=None):
        self.id = id
        self.question = question
        self.label = label

class NerInputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class preProcessorNer:
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
        return ["O", "B-LOC", "I-LOC"]

    def _create_examples(self, path):
        lines = []
        max_len = 0

        with codecs.open(path, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []

            for line in f:
                tokens = line.strip().split()

                if 2 == len(tokens): #  "你 O"
                    word = tokens[0]
                    label = tokens[1]
                    word_list.append(word)
                    label_list.append(label)
                
                elif not tokens: #end of one sentence
                    if len(label_list) > max_len:
                        max_len = len(label_list)

                    lines.append((word_list,label_list))
                    word_list = []
                    label_list = []

        examples = []
        for i, (sentence, label) in enumerate(lines):
            examples.append(
                NerInputText(id=i, question=" ".join(sentence), label=label)
            )
        f.close()
        return examples


def bertEncodeNer(texts, tokenizer, max_seq_length=512, label_list=None):

    all_tokens = []
    all_masks = []
    all_segments = []
    features = []
    
    for text in texts:
        tokens = tokenizer.tokenize(text.question)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        # cls + ids + sep 
        input_ids = tokenizer.build_inputs_with_special_tokens(ids)
        masks = [1] * len(input_ids)
        #token_type_ids = tokenizer.create_token_type_ids_from_sequences(idsA, idsB)
        token_type_ids = [0] * (len(ids) + 2)
        
        # pad seq to max length
        pad_seq = [0] * (max_seq_length - len(input_ids))
        input_ids += pad_seq
        masks += pad_seq
        token_type_ids += pad_seq

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(masks) == max_seq_length, "Error with input length {} vs {}".format(len(masks), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids), max_seq_length)

        # [CLS]/[SEP]: 'O', pad 0 = 'O', keep same lens with other embeddings
        labels_ids = [0] + [label_list.index(i) for i in text.label] + [0] + pad_seq
        assert len(labels_ids) == max_seq_length, "Error with input length {} vs {}".format(len(labels_ids), max_seq_length)

        features.append(
            NerInputFeatures(input_ids, masks, token_type_ids, labels_ids)
        )

    return features


def load_and_cache_example(args, tokenizer, processor, data_type):

    doc_list = ['train','dev','test']

    if data_type not in doc_list:
        raise ValueError("data_type must be one of {}".format(" ".join(doc_list)))

    cached_features_file = "cached_{}_{}".format(data_type, str(args["max_seq_length"]))
    cached_features_file = os.path.join(args["data_dir"], cached_features_file)

    # 加载已处理的feature文件
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

        features = bertEncodeNer(texts=examples, tokenizer=tokenizer, max_seq_length=args["max_seq_length"], label_list=label_list)

        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)

    return dataset


def evaluate_and_save_model(args, model, val_dataloader, best_f1):
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    res = evaluate(args, model, val_dataloader)

    precision_b = res['1']['precision']
    recall_b = res['1']['recall']
    f1_b = res['1']['f1-score']
    support_b = res['1']['support']

    precision_i = res['2']['precision']
    recall_i = res['2']['recall']
    f1_i = res['2']['f1-score']
    support_i = res['2']['support']

    weight_b = support_b / (support_b + support_i)
    weight_i = 1 - weight_b

    avg_precision = precision_b * weight_b + precision_i * weight_i
    avg_recall = recall_b * weight_b + recall_i * weight_i
    avg_f1 = f1_b * weight_b + f1_i * weight_i

    all_avg_precision = res['macro avg']['precision']
    all_avg_recall = res['macro avg']['recall']
    all_avg_f1 = res['macro avg']['f1-score']

    print("[B-LOC] Precision : {:.4f}   Recall : {:.4f}   F1 : {:.4f}   Support : {}"
          .format(precision_b, recall_b, f1_b, support_b))
    print("[I-LOC] Precision : {:.4f}   Recall : {:.4f}   F1 : {:.4f}   Support : {}"
          .format(precision_i, recall_i, f1_i, support_i))
    print("AVG:  Precision : {:.4f}   Recall : {:.4f}   F1 : {:.4f}"
          .format(avg_precision, avg_recall, avg_f1))
    print("all AVG:  Precision : {:.4f}   Recall : {:.4f}   F1 : {:.4f}"
          .format(all_avg_precision, all_avg_recall, all_avg_f1))

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        #model_to_save = model.module if hasattr(model, 'module') else model  
        #model_to_save.save_pretrained(args["output_dir"])
        torch.save(model.state_dict(), os.path.join(args["output_dir"], "pytorch_model.bin"))
        print("save the best model with avg_f1= {:.4f}".format(best_f1))

    return best_f1


def evaluate(args, model, val_dataloader):

    print("--------------- Validation ---------------")
    print("  Num examples = {}".format(len(val_dataloader)))
    print("  Batch size = {}".format(args["batch_size"]))

    all_real_labels = []    
    all_pred_labels = []   
    total_loss = []
    for batch in val_dataloader:
        model.eval()
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode':True,
                      'reduction': 'None'
            }

            outputs = model(**inputs)

            # loss: (batch_size, ) 
            # logits=list[list(int)]: [[00012200],[001222200],..] (batch_size,)
            loss, logits = outputs[0], outputs[1]

            total_loss.extend(loss.tolist()) # tensor -> numpy

            #logits = logits.detach().cpu().numpy()
            label_ids = batch[3].to('cpu').numpy()
            masks = batch[1].to('cpu').numpy()

            all_pred_labels.extend(logits)
            all_real_labels.extend(calc_real_sentences(label_ids, masks, logits))


    total_loss = np.array(total_loss).mean()

    all_real_labels = np.array([i for j in all_real_labels for i in j])
    all_pred_labels = np.array([i for j in all_pred_labels for i in j])
    assert all_real_labels.shape == all_pred_labels.shape
    print(all_real_labels, all_pred_labels)
    res = classification_report(y_true = all_real_labels, y_pred = all_pred_labels, output_dict=True)
    
    return res


def train(args, train_dataloader, val_dataloader, model):

    # optimizer parameter
    gradient_accumulation_steps = 1
    total_lens = len(train_dataloader) // gradient_accumulation_steps * args["epochs"]

    no_decay = ['bias', 'LayerNorm.weight', 'transitions']
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
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args["device"]) for t in batch)
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode': True,
            }
            outputs = model(**inputs)

            loss, logits = outputs[0], outputs[1]

            # 梯度累积计算，当gpu过小
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

        train_acc = evaluate_and_save_model(args, model, val_dataloader, train_acc)
        print("                  avg_loss : {:.4f} ".format(tr_loss/len(train_dataloader)))    

    # 最后循环结束 再评估一次
    train_acc = evaluate_and_save_model(args, model, val_dataloader, train_acc)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_test = {"data_dir": '/content/gdrive/My Drive/nlpqa2/input/data/ner_data/',
                 "load_path": '/content/gdrive/My Drive/nlpqa2/output/data2/',
                 #"vocab_file": 'bert-base-chinese-vocab.txt'
                 #"model_config": 'pytorch_model.bin',
                 #"model_path": 'config.json',
                 "max_seq_length": 128,
                 "batch_size": 256,
                 "learning_rate": 6e-6,
                 "epochs": 3,
                 "device": device
                }
    
    processor = preProcessorNer()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', return_tensors='pt')  

    test_dataset = load_and_cache_example(args_test, tokenizer, processor, 'test')
    model_kwargs = {'config_name': 'bert-base-chinese', 
                    'num_tags':len(processor.get_labels()),
                    'batch_first':True
                    }
    model = bertcrf.BertCrf(**model_kwargs)
    model.load_state_dict(torch.load(os.path.join(args_test["load_path"], "pytorch_model.bin")))
    model = model.to(device)

    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args_test["batch_size"])

    del test_dataset 
    gc.collect()


    all_real_labels = []    
    all_pred_labels = []   
    total_loss = []

    for batch in test_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode':True,
                      'reduction': 'None'
            }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            total_loss.extend(loss.tolist()) # tensor -> numpy
            label_ids = batch[3].to('cpu').numpy()
            masks = batch[1].to('cpu').numpy()

            all_pred_labels.extend(logits)
            all_real_labels.extend(calc_real_sentences(label_ids, masks, logits))

    total_loss = np.array(total_loss).mean()

    all_real_labels = np.array([i for j in all_real_labels for i in j])
    all_pred_labels = np.array([i for j in all_pred_labels for i in j])
    assert all_real_labels.shape == all_pred_labels.shape

    res = classification_report(y_true = all_real_labels, y_pred = all_pred_labels, output_dict=True)
    print(res)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device", device)

    args_train = {"data_dir": '/content/gdrive/My Drive/nlpqa2/input/data/ner_data/',
            "pre_train_model": 'bert-base-chinese',
            "output_dir": '/content/gdrive/My Drive/nlpqa2/output/data2/',
            "max_seq_length": 128,
            "batch_size": 16,
            "learning_rate": 6e-6,
            "epochs": 3,
            "device": device
    }

    assert os.path.exists(args_train["data_dir"])

    processor = preProcessorNer()

    tokenizer = BertTokenizer.from_pretrained(args_train["pre_train_model"], return_tensors='pt')  

    train_dataset = load_and_cache_example(args_train, tokenizer, processor, 'train')
    val_dataset = load_and_cache_example(args_train, tokenizer, processor, 'dev')
    
    model_kwargs = {'config_name': args_train["pre_train_model"], 
                    'model_name':args_train["pre_train_model"], 
                    'num_tags':len(processor.get_labels()),
                    'batch_first':True
                    }
    model = bertcrf.BertCrf(**model_kwargs)
    model = model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=args_train["batch_size"], sampler=RandomSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args_train["batch_size"])
    
    del train_dataset, val_dataset 
    gc.collect()

    train(args_train, train_dataloader, val_dataloader, model)
    
    del train_dataloader, val_dataloader
    gc.collect()
    #test()