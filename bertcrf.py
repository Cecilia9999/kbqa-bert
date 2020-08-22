# -*- coding: utf-8 -*-

from typing import List, Optional
import os
import torch
import torch.nn as nn

from transformers import BertForTokenClassification,BertTokenizer,BertConfig


# CRF Ref: https://pytorch-crf.readthedocs.io/en/stable/index.html
class CRF(nn.Module):
	
    def __init__(self, num_tags : int = 2, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # score of start-> 
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        
        # score of ->tag_end 
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        
        # transitions[i][j]: score of tag_i -> tag_j
        self.transitions = nn.Parameter(torch.empty(num_tags,num_tags))

        self.reset_parameters()


    def reset_parameters(self):
        init_range = 0.1
        nn.init.uniform_(self.start_transitions, -init_range, init_range)
        nn.init.uniform_(self.end_transitions, -init_range, init_range)
        nn.init.uniform_(self.transitions, -init_range, init_range)


    def __repr__(self):
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    
    # Compute the conditional log likelihood of a sequence of tags given emission scores
    # change to negative log likelihood
    # p(y|x) = exp(Score(X, y)) / sum(exp(Score(X, y)))
    # log P = Score(X, y) - log(sum(exp(Score(X, y))))
    # loss = - log P = log(sum(exp(Score(X, y)))) - Score(X, y)
    def forward(self, 
                emissions:torch.Tensor,
                tags:torch.Tensor = None,
                mask:Optional[torch.ByteTensor] = None,
                reduction: str = "mean") -> torch.Tensor:

        self._validate(emissions, tags = tags ,mask = mask)

        reduction = reduction.lower()
        if reduction not in ('none','sum','mean','token_mean'):
            raise ValueError(f'invalid reduction {reduction}')

        if mask is None:
            mask = torch.ones_like(tags,dtype = torch.uint8)

        if self.batch_first:
            # emissions.shape (seq_len,batch_size,tag_num)
            emissions = emissions.transpose(0,1)
            tags = tags.transpose(0,1)
            mask = mask.transpose(0,1)

        # shape: (batch_size,)
        numerator = self._computer_score(emissions=emissions,tags=tags,mask=mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions=emissions,mask=mask)
        # shape: (batch_size,)
        nllh = denominator - numerator

        if reduction == 'none':
            return nllh
        elif reduction == 'sum':
            return nllh.sum()
        elif reduction == 'mean':
            return nllh.mean()
        assert reduction == 'token_mean'
        return nllh.sum() / mask.float().sum()


    def decode(self,emissions:torch.Tensor,
               mask : Optional[torch.ByteTensor] = None) ->List[List[int]]:
               
        self._validate(emissions=emissions,mask=mask)

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2],dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0,1)
            mask = mask.transpose(0,1)

        return self._viterbi_decode(emissions,mask)


    def _validate(self,
                  emissions:torch.Tensor,
                  tags:Optional[torch.LongTensor] = None ,
                  mask:Optional[torch.ByteTensor] = None) -> None:

        if emissions.dim() != 3:
            raise ValueError(f"emissions must have dimension of 3 , got {emissions.dim()}")
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags},'
                f'got {emissions.size(2)}'
            )

        if tags is not None:

            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of and mask must match,'
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}'
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:,0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')


    def _computer_score(self,
                        emissions:torch.Tensor,
                        tags:torch.LongTensor,
                        mask:torch.ByteTensor) -> torch.Tensor:

        # batch second
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()


        # Start transition score and first emission
        # self.start_transitions:  score of start-> 
        # score.shape: (batch_size, )
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size, )
            score += self.transitions[tags[i-1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]


        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # final tags of each sample
        last_tags = tags[seq_ends, torch.arange(batch_size)]

        # shape: (batch_size,) 每一个样本到最后一个词的得分加上之前的score
        score += self.end_transitions[last_tags]

        return score


    def _compute_normalizer(self,
                            emissions:torch.Tensor ,
                            mask: torch.ByteTensor) -> torch.Tensor:

        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # shape : (batch_size,num_tag)
        # self.start_transitions  start 到其他tag(不包含end)的得分
        # start_transitions.shape tag_nums     emissions[0].shape (batch_size,tag_size)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1,seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(dim=2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score,dim = 1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score,dim=1)


    def _viterbi_decode(self, 
                        emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class BertCrf(nn.Module):
    def __init__(self, 
                 config_name:str = 'bert-base-chinese', 
                 model_name:str = None, 
                 num_tags: int = 2, 
                 batch_first:bool = True) -> None:

        # 记录batch_first
        self.batch_first = batch_first

        # 加载模型配置文件
        if config_name != 'bert-base-chinese':
            if not os.path.exists(config_name):
                raise ValueError(
                    "Error! No model config file: '{}'".format(config_name)
                )
            else:
                self.config_name = config_name
        else:
            self.config_name = config_name

        # 加载预训练模型
        if model_name is not None:
            if model_name == 'bert-base-chinese':
                self.model_name = model_name
            elif not os.path.exists(model_name):
                raise ValueError(
                    "Error! No pretrained model: '{}'".format(model_name)
                )
            else:
                self.model_name = model_name
        else:
            self.model_name = None

        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')

        super().__init__()

        self.bert_config = BertConfig.from_pretrained(self.config_name)
        self.bert_config.num_labels = num_tags

        # 如果模型不存在
        if self.model_name is None:
            self.model_kwargs = {'config': self.bert_config}
            self.bertModel = BertForTokenClassification(**self.model_kwargs)
        elif self.model_name == 'bert-base-chinese':
            self.model_kwargs = {'config': self.bert_config, "from_tf": True}
            self.bertModel = BertForTokenClassification.from_pretrained(self.model_name, **self.model_kwargs)

        self.crf_model = CRF(num_tags=num_tags, batch_first=batch_first)


    def forward(self,input_ids:torch.Tensor,
                tags:torch.Tensor = None,
                attention_mask:Optional[torch.ByteTensor] = None,
                token_type_ids=torch.Tensor,
                decode:bool = True,       # 是否预测编码
                reduction: str = "mean") -> List:

        out = self.bertModel(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids)
        emissions = out[0]
        # 这里在seq_len的维度上去头，是去掉了[CLS]，去尾巴有两种情况
        # 1、是 <pad> 2、[SEP]

        new_emissions = emissions[:, 1:-1] # del [CLS], [SEP]
        new_mask = attention_mask[:,2:].bool()

        # tags=None -> prediction, no loss
        if tags == None:
            loss = None
            pass
        else:
            new_tags = tags[:, 1:-1]
            loss = self.crf_model(emissions=new_emissions, tags=new_tags, mask=new_mask, reduction=reduction)

        if decode:
            tag_list = self.crf_model.decode(emissions=new_emissions, mask=new_mask)
            return [loss, tag_list]

        return [loss]

