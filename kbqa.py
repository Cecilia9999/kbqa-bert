# -*- coding: utf-8 -*-

import sys
import os
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import bertcrf, ner_main, sim_main

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

import numpy as np
import pandas as pd

#import pymysql
from tqdm import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_ner_model(config_file, model_name, label_num=3):
    model = bertcrf.BertCrf(config_name=config_file, num_tags=label_num, batch_first=True)
    model.load_state_dict(torch.load(model_name))
    return model


# model_name=os.path.join(args_test["load_path"], 'pytorch_model.bin')
def load_sim_model(config_file, model_name, label_num=2):
    bert_config = BertConfig.from_pretrained(config_file)
    bert_config.num_labels = label_num
    model_kwargs = {'config': bert_config, "from_tf": False}
    model = BertForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    return model


def get_entity(model, tokenizer, sentence, max_seq_length=128):

    sentence_list = list(sentence.strip().replace(' ',''))
    text = " ".join(sentence_list)

    tokens = tokenizer.tokenize(text)
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


    input_ids = torch.tensor(input_ids).reshape(1, -1).to(device)
    masks = torch.tensor(masks).reshape(1, -1).to(device)
    token_type_ids = torch.tensor(token_type_ids).reshape(1, -1).to(device)

    model = model.to(device)
    model.eval()
    
    # when label is None, loss won't be computed
    output = model(input_ids = input_ids,
                tags = None,
                attention_mask = masks,
                token_type_ids = token_type_ids,
                decode=True)
    
    pred_tag = output[1][0]
    assert len(pred_tag) == len(sentence_list) or len(pred_tag) == max_len - 2

    pred_tag_len = len(pred_tag)

    CRF_LABELS = ['O', 'B-LOC', 'I-LOC']
    b_loc_idx = CRF_LABELS.index('B-LOC')
    i_loc_idx = CRF_LABELS.index('I-LOC')
    o_idx = CRF_LABELS.index('O')

    if b_loc_idx not in pred_tag and i_loc_idx not in pred_tag:
        print("没有在句子[{}]中发现实体".format(sentence))
        return ''
    if b_loc_idx in pred_tag:
        entity_start_idx = pred_tag.index(b_loc_idx)
    else:
        entity_start_idx = pred_tag.index(i_loc_idx)

    entity_list = []
    entity_list.append(sentence_list[entity_start_idx])

    for i in range(entity_start_idx+1, pred_tag_len):
        if pred_tag[i] == i_loc_idx:
            entity_list.append(sentence_list[i])
        else:
            break

    return "".join(entity_list)


def semantic_match(model, tokenizer, question, attribute_list, answer_list, max_seq_length):

    assert len(attribute_list) == len(answer_list)

    all_tokens = []
    all_masks = []
    all_segments = []
    features = []
    for attribute in attribute_list:

        textA = tokenizer.tokenize(question)
        textB = tokenizer.tokenize(attribute)
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
            sim_main.SimInputFeatures(input_ids, masks, token_type_ids)
        )

       
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    assert all_input_ids.shape == all_attention_mask.shape
    assert all_attention_mask.shape == all_token_type_ids.shape


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=128)

    data_num = all_attention_mask.shape[0]
    batch_size = 128
    
    all_logits = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': None
                     }

            outputs = model(**inputs)
            
            logits = outputs[0]

            #logits = logits.sigmoid(dim = -1)
            #logits = logits.softmax(dim = -1)

            if all_logits is None:
                all_logits = logits.clone()
            else:
                all_logits = torch.cat([all_logits, logits], dim = 0)

    prediction = all_logits.argmax(dim = -1)
    if prediction.sum() == 0:
        return torch.tensor(-1)
    else:
        return prediction.argmax(dim = -1)


def select_database(sql):
    # connect database
    connect = pymysql.connect(user="root", password="123456", host="127.0.0.1", port=3306, db="kb_qa", charset="utf8")
    cursor = connect.cursor()  # 创建操作游标
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
    except Exception as e:
        print("Error: unable to fecth data: %s ,%s" % (repr(e), sql))
    finally:
        # 关闭数据库连接
        cursor.close()
        connect.close()
    return results


# directly match: check if attribute is in question 
def text_match(attribute_list, answer_list, sentence):

    assert len(attribute_list) == len(answer_list)

    idx = -1
    for i, attribute in enumerate(attribute_list):
        if attribute in sentence:
            idx = i
            break

    if idx != -1:
        return attribute_list[idx], answer_list[idx]
    
    return "",""


def main():

    with torch.no_grad():
        ner_path = '/content/gdrive/My Drive/nlpqa2/output/data2/'
        sim_path = '/content/gdrive/My Drive/nlpqa2/output/data/'
        ner_processor = ner_main.preProcessorNer()
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', return_tensors='pt')

        ner_model = load_ner_model(config_file='bert-base-chinese',
                                   model_name=os.path.join(ner_path, 'pytorch_model.bin'), label_num=len(ner_processor.get_labels()))
        
        ner_model = ner_model.to(device)
        ner_model.eval()


        while True:
            print("====="*8)
            raw_text = input("Please input the question(quit: 'q')：\n")
            raw_text = raw_text.strip()
            if (raw_text == 'q'):
                print("Thank U for using KBQA!")
                return
            
            entity = get_entity(model=ner_model, tokenizer=tokenizer, sentence=raw_text, max_seq_length=128)
            print("Entity: ", entity)
            
            if not entity:
                print("Cannot find entity in question!")
                continue
            
            # search entity via database
            #sql_str = "select * from nlpccqa where entity = '{}'".format(entity)
            #triple_list = select_database(sql_str)

            # search entity from clean_triple file directly
            df = pd.read_csv('/content/gdrive/My Drive/nlpqa2/input/data/DB_Data/clean_triple.csv')

            triple_list = []
            pick = df.loc[df['entity'] == entity]
            triple_list = pick.values.tolist()
            
            if len(triple_list) == 0:
                print("There is no related info about entity '{}'".format(entity))
                continue
            
            print('find triple list:\n', triple_list)
            triple_list = list(zip(*triple_list))
            print('attr list:\n', triple_list[1])
            
            attribute_list = triple_list[1]
            answer_list = triple_list[2]
            attribute, answer = text_match(attribute_list, answer_list, raw_text)

            if attribute and answer:
                res = "{}的{}是{}".format(entity, attribute, answer)
            else:
                sim_processor = sim_main.preProcessor()
                sim_model = load_sim_model(config_file=os.path.join(sim_path, 'config.json'),
                                           model_name=os.path.join(sim_path, 'pytorch_model.bin'),
                                           label_num=len(sim_processor.get_labels()))

                sim_model = sim_model.to(device)
                sim_model.eval()

                attribute_idx = semantic_match(sim_model, tokenizer, raw_text, attribute_list, answer_list, 64).item()
                
                if attribute_idx == -1:
                    res = ''
                else:
                    attribute = attribute_list[attribute_idx]
                    answer = answer_list[attribute_idx]
                    res = "{}的{}是{}".format(entity, attribute, answer)

            if res == '':
                print("There is no answer about entity '{}'".format(entity))
            else:
                print("Answer:", res)


if __name__ == '__main__':
    main()