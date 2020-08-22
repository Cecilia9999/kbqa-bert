# kbqa-bert
construct simple kbqa system based on deep learning(BERT)

## 数据
input/data
NLPCC2016的中文KBQA数据集

【original data】
http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html

【处理过的三元组数据集】
https://github.com/huangxiangzhou/NLPCC2016KBQA 

## 数据预处理
- split_dataset.py: 将数据集分为train(14609)/dev(9870)/test(9870)三部分

- ner-dataset2.py: 构造NER数据集，采用BIO标签

- sim-dataset3.py: 构造【问题-属性】数据集，二分类问题：1个问题-正确实体：label=1 / 5个问题-错误实体：label=0

- triple_clean4.py: 构造三元组数据集，存为csv文件，用于mention查询

## 训练模型1
- ner_main.py: 基于pytorch crf + bert，用于识别输入问题中的mention

crf（ref：https://pytorch-crf.readthedocs.io/en/stable/index.html#）

bertcrf（ref：https://github.com/997261095/bert-kbqa/blob/master/BERT_CRF.py）

## 训练模型2
- sim_main.py: 基于 pytorch bert，用于计算问题和属性的相似度，判断问题是否包含该属性

## 综合以上两个模型
- kbqa.py: 
- 1）对于输入问题，利用NER模型识别问题中的 mention；
- 2）简单检索 triple_clean.csv，进行实体链接，找出知识库中所有对应的 entity 及其对应候选 attribute 和 answer；
- 3）简单字符串匹配：若候选属性出现在问题中，则问题-属性直接匹配成功，返回对应 answer；
- 4）问题-属性匹配：若候选属性没有直接出现在问题中，利用问题-属性相似度模型进行预测，返回对应 answer

## 有待改进
- 1）Candidate Entity Generation：同一 entity 存在不同 mention
- 2）Entity Disambiguation： 同一 mention 对应不同 entity

