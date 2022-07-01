# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import json
import logging
import random
import argparse

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

from metrics import tag_mapping_nearest, tag_mapping_corres
from utils import Label2IdxSub, Label2IdxObj
import utils
from dataloader import CustomDataLoader

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--mode', type=str, default="test")
parser.add_argument('--corpus_type', type=str, default="Job", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='last', help="name of the file containing weights to reload")

parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")


def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    TP = correct_num
    FP = predict_num - correct_num
    FN = gold_num - correct_num
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output

def span2str_revised(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result
    with open(params.data_dir / 'rel2id.json', 'r', encoding='utf-8') as fr:
        id2rel = json.load(fr)[0]
    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append(" | ".join([sub, id2rel[str(rel)], obj]))

    return output


def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    rel_num = params.rel_num

    predictions = []
    ground_truths = []
    # TP/FP/FN calculation 
    
    TP_ks, TP_ka, TP_es, TP_ea, TP_de = [], [], [], [], []
    FP_ks, FP_ka, FP_es, FP_ea, FP_de = [], [], [], [], []
    FN_ks, FN_ka, FN_es, FN_ea, FN_de = [], [], [], [], []

    TP_all = (TP_ks, TP_ka, TP_es, TP_ea, TP_de)
    FP_all = (FP_ks, FP_ka, FP_es, FP_ea, FP_de)
    FN_all = (FN_ks, FN_ka, FN_es, FN_ea, FN_de)

    # classification report for relation 
    correct_num, predict_num, gold_num = 0, 0, 0
    correct_ks, predict_ks, gold_ks = 0, 0, 0
    correct_ka, predict_ka, gold_ka = 0, 0, 0
    correct_es, predict_es, gold_es = 0, 0, 0
    correct_ea, predict_ea, gold_ea = 0, 0, 0
    correct_d, predict_d, gold_d = 0, 0, 0

    # classification report for entities

    correct_k, predict_k, gold_k = 0, 0, 0
    correct_a, predict_a, gold_a = 0, 0, 0
    correct_s, predict_s, gold_s = 0, 0, 0
    correct_e, predict_e, gold_e = 0, 0, 0
    correct_di, predict_di, gold_di = 0, 0, 0
    correct_ma, predict_ma, gold_ma = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)

            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            if ex_params['ensure_rel']:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            gold_triples = span2str(triples[idx], input_tokens[idx])
         
            pre_triples = span2str(pre_triples, input_tokens[idx])
            print('gold:  ', gold_triples)
            print('pre:  ', pre_triples)

            knowledge_skills_gold = [i for i in gold_triples if i[-1] ==0]
            knowledge_area_gold = [i for i in gold_triples if i[-1] ==1]
            experience_skills_gold = [i for i in gold_triples if i[-1] ==2]
            experience_area_gold = [i for i in gold_triples if i[-1] ==3]
            degree_in_gold = [i for i in gold_triples if i[-1] ==4]

            knowledge_skills_pre = [i for i in pre_triples if i[-1] ==0]
            knowledge_area_pre = [i for i in pre_triples if i[-1] ==1]
            experience_skills_pre = [i for i in pre_triples if i[-1] ==2]
            experience_area_pre = [i for i in pre_triples if i[-1] ==3]
            degree_in_pre = [i for i in pre_triples if i[-1] ==4]

            print(set(pre_triples) & set(gold_triples))

            # ground_truths.append(list(set(gold_triples)))
            # predictions.append(list(set(pre_triples)))

            ground_truths.append(gold_triples)
            predictions.append(pre_triples)
            # counter
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))

            # counter_knowledge_skills
            correct_ks += len(set(knowledge_skills_pre) & set(knowledge_skills_gold))
            predict_ks += len(set(knowledge_skills_pre))
            gold_ks += len(set(knowledge_skills_gold))

            """error analysis for knowledge_skills"""
            tp_ks = set(knowledge_skills_pre) & set(knowledge_skills_gold)
            fp_ks = set(knowledge_skills_pre).difference(set(tp_ks))
            fn_ks = set(knowledge_skills_gold).difference(set(tp_ks))

            """append examples for knowledge_skills"""
            TP_ks.append(tp_ks)
            FP_ks.append(fp_ks)
            FN_ks.append(fn_ks)

            # counter_knowledge_areas
            correct_ka += len(set(knowledge_area_pre) & set(knowledge_area_gold))
            predict_ka += len(set(knowledge_area_pre))
            gold_ka += len(set(knowledge_area_gold))

            """error analysis for knowledge_areas"""
            tp_ka = set(knowledge_area_pre) & set(knowledge_area_gold)
            fp_ka = set(knowledge_area_pre).difference(set(tp_ka))
            fn_ka = set(knowledge_area_gold).difference(set(tp_ka))
            
            """append examples for knowledge_areas"""
            TP_ka.append(tp_ka)
            FP_ka.append(fp_ka)
            FN_ka.append(fn_ka)

            # counter_experience_skills
            correct_es += len(set(experience_skills_pre) & set(experience_skills_gold))
            predict_es += len(set(experience_skills_pre))
            gold_es += len(set(experience_skills_gold))
            
            """error analysis for experience_skills"""
            tp_es = set(experience_skills_pre) & set(experience_skills_gold)
            fp_es = set(experience_skills_pre).difference(set(tp_es))
            fn_es = set(experience_skills_gold).difference(set(tp_es))

            """append examples for experience_skills"""
            TP_es.append(tp_es)
            FP_es.append(fp_es)
            FN_es.append(fn_es)

            # counter_experience_areas
            correct_ea += len(set(experience_area_pre) & set(experience_area_gold))
            predict_ea += len(set(experience_area_pre))
            gold_ea += len(set(experience_area_gold))

            """error analysis for experience_areas"""
            tp_ea = set(experience_area_pre) & set(experience_area_gold)
            fp_ea = set(experience_area_pre).difference(set(tp_ea))
            fn_ea = set(experience_area_gold).difference(set(tp_ea))
            
            """append examples for experience_areas"""
            TP_ea.append(tp_ea)
            FP_ea.append(fp_ea)
            FN_ea.append(fn_ea)

            # counter_degree_in
            correct_d += len(set(degree_in_gold) & set(degree_in_pre))
            predict_d += len(set(degree_in_pre))
            gold_d += len(set(degree_in_gold))

            """error analysis for degree_in"""
            tp_de = set(degree_in_pre) & set(degree_in_gold)
            fp_de = set(degree_in_pre).difference(set(tp_de))
            fn_de = set(degree_in_gold).difference(set(tp_de))

            """append examples for experience_areas"""
            TP_de.append(tp_de)
            FP_de.append(fp_de)
            FN_de.append(fn_de)

            #####################################################

            # counter_knowledge entity & counter_skills entity
            correct_k += len(set(knowledge_skills_pre) & set(knowledge_skills_gold))
            predict_k += len(set(knowledge_skills_pre))
            gold_k += len(set(knowledge_skills_gold))

            correct_s += len(set(knowledge_skills_pre) & set(knowledge_skills_gold))
            predict_s += len(set(knowledge_skills_pre))
            gold_s += len(set(knowledge_skills_gold))

             # counter_knowledge entity & counter_area entity
            correct_k += len(set(knowledge_area_pre) & set(knowledge_area_gold))
            predict_k += len(set(knowledge_area_pre))
            gold_k += len(set(knowledge_area_gold))

            correct_a += len(set(knowledge_area_pre) & set(knowledge_area_gold))
            predict_a += len(set(knowledge_area_pre))
            gold_a += len(set(knowledge_area_gold))

            # counter_experience entity & counter_skills entity
            correct_e += len(set(experience_skills_pre) & set(experience_skills_gold))
            predict_e += len(set(experience_skills_pre))
            gold_e += len(set(experience_skills_gold))

            correct_s += len(set(experience_skills_pre) & set(experience_skills_gold))
            predict_s += len(set(experience_skills_pre))
            gold_s += len(set(experience_skills_gold))

            # counter_experience entity & counter_areas entity
            correct_e += len(set(experience_area_pre) & set(experience_area_gold))
            predict_e += len(set(experience_area_pre))
            gold_e += len(set(experience_area_gold))

            correct_a += len(set(experience_area_pre) & set(experience_area_gold))
            predict_a += len(set(experience_area_pre))
            gold_a += len(set(experience_area_gold))

            # counter_diploma entity & counter major entity

            correct_di += len(set(degree_in_gold) & set(degree_in_pre))
            predict_di += len(set(degree_in_pre))
            gold_di += len(set(degree_in_gold))

            correct_ma += len(set(degree_in_gold) & set(degree_in_pre))
            predict_ma += len(set(degree_in_pre))
            gold_ma += len(set(degree_in_gold))


    metrics = get_metrics(correct_num, predict_num, gold_num)

    """metrics for relation:"""
    ############################# 
    metrics_ks = get_metrics(correct_ks, predict_ks, gold_ks)
    metrics_ka = get_metrics(correct_ka, predict_ka, gold_ka)
    metrics_es = get_metrics(correct_es, predict_es, gold_es)
    metrics_ea = get_metrics(correct_ea, predict_ea, gold_ea)
    metrics_d = get_metrics(correct_d, predict_d, gold_d)

    """metrics for entities"""
    ##############################

    metrics_k = get_metrics(correct_k, predict_k, gold_k)
    metrics_a = get_metrics(correct_a, predict_a, gold_a)
    metrics_s = get_metrics(correct_s, predict_s, gold_s)
    metrics_e = get_metrics(correct_e, predict_e, gold_e)
    metrics_di = get_metrics(correct_di, predict_di, gold_di)
    metrics_ma = get_metrics(correct_ma, predict_ma, gold_ma)

    # logging loss, f1 and report for relation 
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    metrics_str_ks = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ks.items())
    metrics_str_ka = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ka.items())
    metrics_str_es = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_es.items())
    metrics_str_ea = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ea.items())
    metrics_str_d = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_d.items())

    # logging loss, f1 and report for entity 

    metrics_str_k = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_k.items())
    metrics_str_a = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_a.items())
    metrics_str_s = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_s.items())
    metrics_str_e = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_e.items())
    metrics_str_di = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_di.items())
    metrics_str_ma = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ma.items())

    ####################### logging for relation
    logging.info("- {} metrics- All:\n".format(mark) + metrics_str)

    logging.info("- {} metrics- knowledge_skills:\n".format(mark) + metrics_str_ks)
    logging.info("- {} metrics- knowledge_areas:\n".format(mark) + metrics_str_ka)
    logging.info("- {} metrics- experience_skills:\n".format(mark) + metrics_str_es)
    logging.info("- {} metrics- experience_areas:\n".format(mark) + metrics_str_ea)
    logging.info("- {} metrics- degree_in :\n".format(mark) + metrics_str_d)

    ####################### logging for entities

    logging.info("- {} metrics- knowledge:\n".format(mark) + metrics_str_k)
    logging.info("- {} metrics- areas:\n".format(mark) + metrics_str_a)
    logging.info("- {} metrics- skills:\n".format(mark) + metrics_str_s)
    logging.info("- {} metrics- experience:\n".format(mark) + metrics_str_e)
    logging.info("- {} metrics- diploma :\n".format(mark) + metrics_str_di)
    logging.info("- {} metrics- major :\n".format(mark) + metrics_str_ma)


    return metrics, predictions, ground_truths, metrics_ks, metrics_ka, metrics_es, metrics_ea, metrics_d, metrics_k, metrics_a,metrics_s, metrics_e, metrics_di, metrics_ma, TP_all, FP_all, FN_all


def evaluate_revised(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    rel_num = params.rel_num

    predictions = []
    ground_truths = []

    predictions_revised = []
    ground_truths_revised = []

    TP_ks, TP_ka, TP_es, TP_ea, TP_de = [], [], [], [], []
    FP_ks, FP_ka, FP_es, FP_ea, FP_de = [], [], [], [], []
    FN_ks, FN_ka, FN_es, FN_ea, FN_de = [], [], [], [], []

    TP_all = (TP_ks, TP_ka, TP_es, TP_ea, TP_de)
    FP_all = (FP_ks, FP_ka, FP_es, FP_ea, FP_de)
    FN_all = (FN_ks, FN_ka, FN_es, FN_ea, FN_de)
   
    # classification report for relation 
    correct_num, predict_num, gold_num = 0, 0, 0
    correct_ks, predict_ks, gold_ks = 0, 0, 0
    correct_ka, predict_ka, gold_ka = 0, 0, 0
    correct_es, predict_es, gold_es = 0, 0, 0
    correct_ea, predict_ea, gold_ea = 0, 0, 0
    correct_d, predict_d, gold_d = 0, 0, 0

    # classification report for entities

    correct_k, predict_k, gold_k = 0, 0, 0
    correct_a, predict_a, gold_a = 0, 0, 0
    correct_s, predict_s, gold_s = 0, 0, 0
    correct_e, predict_e, gold_e = 0, 0, 0
    correct_di, predict_di, gold_di = 0, 0, 0
    correct_ma, predict_ma, gold_ma = 0, 0, 0
    

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)

            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            if ex_params['ensure_rel']:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            """revised output as input for text generation"""
            gold_triples_revised = span2str_revised(triples[idx], input_tokens[idx])
            pre_triples_revised = span2str_revised(pre_triples, input_tokens[idx])

            """original output as for evaluation"""
            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])

            # ground_truths_revised.append(list(set(gold_triples_revised)))
            # predictions_revised.append(list(set(pre_triples_revised)))

            ground_truths_revised.append(gold_triples_revised)
            predictions_revised.append(pre_triples_revised)


            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))

            """categorize relations in different groups for ground truths"""
            knowledge_skills_gold = [i for i in gold_triples if i[-1] ==0]
            knowledge_area_gold = [i for i in gold_triples if i[-1] ==1]
            experience_skills_gold = [i for i in gold_triples if i[-1] ==2]
            experience_area_gold = [i for i in gold_triples if i[-1] ==3]
            degree_in_gold = [i for i in gold_triples if i[-1] ==4]

            """categorize relations in different groups for predictions"""
            knowledge_skills_pre = [i for i in pre_triples if i[-1] ==0]
            knowledge_area_pre = [i for i in pre_triples if i[-1] ==1]
            experience_skills_pre = [i for i in pre_triples if i[-1] ==2]
            experience_area_pre = [i for i in pre_triples if i[-1] ==3]
            degree_in_pre = [i for i in pre_triples if i[-1] ==4]

            # counter
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))

            # counter_knowledge_skills
            correct_ks += len(set(knowledge_skills_pre) & set(knowledge_skills_gold))
            predict_ks += len(set(knowledge_skills_pre))
            gold_ks += len(set(knowledge_skills_gold))

            """error analysis for knowledge_skills"""
            tp_ks = set(knowledge_skills_pre) & set(knowledge_skills_gold)
            fp_ks = set(knowledge_skills_pre)
            fn_ks = set(knowledge_skills_gold)

            """append examples for knowledge_skills"""
            TP_ks.append(tp_ks)
            FP_ks.append(fp_ks)
            FN_ks.append(fn_ks)

            # counter_knowledge_areas
            correct_ka += len(set(knowledge_area_pre) & set(knowledge_area_gold))
            predict_ka += len(set(knowledge_area_pre))
            gold_ka += len(set(knowledge_area_gold))

            """error analysis for knowledge_areas"""
            tp_ka = set(knowledge_area_pre) & set(knowledge_area_gold)
            fp_ka = set(knowledge_area_pre)
            fn_ka = set(knowledge_area_gold)
            
            """append examples for knowledge_areas"""
            TP_ka.append(tp_ka)
            FP_ka.append(fp_ka)
            FN_ka.append(fn_ka)

            # counter_experience_skills
            correct_es += len(set(experience_skills_pre) & set(experience_skills_gold))
            predict_es += len(set(experience_skills_pre))
            gold_es += len(set(experience_skills_gold))

            
            """error analysis for experience_skills"""
            tp_es = set(experience_skills_pre) & set(experience_skills_gold)
            fp_es = set(experience_skills_pre)
            fn_es = set(experience_skills_gold)

            """append examples for experience_skills"""
            TP_es.append(tp_es)
            FP_es.append(fp_es)
            FN_es.append(fn_es)

            # counter_experience_areas
            correct_ea += len(set(experience_area_pre) & set(experience_area_gold))
            predict_ea += len(set(experience_area_pre))
            gold_ea += len(set(experience_area_gold))

            """error analysis for experience_areas"""
            tp_ea = set(experience_area_pre) & set(experience_area_gold)
            fp_ea = set(experience_area_pre)
            fn_ea = set(experience_area_gold)
            
            """append examples for experience_areas"""
            TP_ea.append(tp_ea)
            FP_ea.append(fp_ea)
            FN_ea.append(fn_ea)

            # counter_degree_in
            correct_d += len(set(degree_in_gold) & set(degree_in_pre))
            predict_d += len(set(degree_in_pre))
            gold_d += len(set(degree_in_gold))
            
            """error analysis for degree_in"""
            tp_de = set(degree_in_pre) & set(degree_in_gold)
            fp_de = set(degree_in_pre)
            fn_de = set(degree_in_gold)

            """append examples for experience_areas"""
            TP_de.append(tp_de)
            FP_de.append(fp_de)
            FN_de.append(fn_de)

            print(set(pre_triples) & set(gold_triples))


    metrics = get_metrics(correct_num, predict_num, gold_num)

    """metrics for relation:"""
    ############################# 
    metrics_ks = get_metrics(correct_ks, predict_ks, gold_ks)
    metrics_ka = get_metrics(correct_ka, predict_ka, gold_ka)
    metrics_es = get_metrics(correct_es, predict_es, gold_es)
    metrics_ea = get_metrics(correct_ea, predict_ea, gold_ea)
    metrics_d = get_metrics(correct_d, predict_d, gold_d)

    # logging loss, f1 and report for relation 
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    metrics_str_ks = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ks.items())
    metrics_str_ka = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ka.items())
    metrics_str_es = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_es.items())
    metrics_str_ea = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_ea.items())
    metrics_str_d = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_d.items())

    ####################### logging for relation
    logging.info("- {} metrics- All:\n".format(mark) + metrics_str)

    logging.info("- {} metrics- knowledge_skills:\n".format(mark) + metrics_str_ks)
    logging.info("- {} metrics- knowledge_areas:\n".format(mark) + metrics_str_ka)
    logging.info("- {} metrics- experience_skills:\n".format(mark) + metrics_str_es)
    logging.info("- {} metrics- experience_areas:\n".format(mark) + metrics_str_ea)
    logging.info("- {} metrics- degree_in :\n".format(mark) + metrics_str_d)

    return metrics, predictions_revised, ground_truths_revised, predictions, ground_truths, metrics_ks, metrics_ka, metrics_es, metrics_ea, metrics_d, TP_all, FP_all, FN_all

if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        'corres_threshold': args.corres_threshold,
        'rel_threshold': args.rel_threshold,
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'emb_fusion': args.emb_fusion
    }

    # torch.cuda.set_device(args.device_id)
    # print('current device:', torch.cuda.current_device())
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")

    """call evaluation_revised function to measure the confusion matrix and write the result to csv"""

    test_metrics, predictions_revised, ground_truths_revised, predictions, ground_truths, metrics_ks, metrics_ka, metrics_es, metrics_ea, metrics_d, TP_all, FP_all, FN_all = evaluate_revised(model, loader, params, ex_params, mark=mode)

    """write error analysis for relation per cateogory"""

    names = ['knowledge_skills', 'knowledge_area', 'experience_skills', 
          'experience_areas', 'degree_in']
      
    for idx, name in enumerate(names):
      with open('Error_analysis/{}.txt'.format(name),'w', encoding='utf8') as f:
          for _pred in TP_all[idx]:
              for i in _pred:
                f.write(str(i))
                f.write('\n')
          f.write('------Correct relation------')
          f.write('\n')
          for _pred in FP_all[idx]:
              for i in _pred:
                f.write(str(i))
                f.write('\n')
          f.write('------predicted relation------')
          f.write('\n')
          for _pred in FN_all[idx]:
              for i in _pred:
                f.write(str(i))
                f.write('\n')
          f.write('------gold relation------')
          f.write('\n')
    
    #############################
    """confusion matrix for relation"""
    TPa, TPb, TPc, TPd, TPe = metrics_ks['TP'], metrics_ka['TP'], metrics_es['TP'], metrics_ea['TP'], metrics_d['TP']
    FPa, FPb, FPc, FPd, FPe = metrics_ks['FP'], metrics_ka['FP'], metrics_es['FP'], metrics_ea['FP'], metrics_d['FP']
    FNa, FNb, FNc, FNd, FNe = metrics_ks['FN'], metrics_ka['FN'], metrics_es['FN'], metrics_ea['FN'], metrics_d['FN']
    Micro_average_p = (TPa + TPb + TPc + TPd + TPe) /  (TPa + TPb + TPc + TPd + TPe + FPa +FPb + FPc + FPd + FPe) if (TPa + TPb + TPc + TPd + TPe + FPa +FPb + FPc + FPd + FPe) > 0 else 0
    Micro_average_r = (TPa + TPb + TPc + TPd + TPe) /  (TPa + TPb + TPc + TPd + TPe + FNa +FNb + FNc + FNd + FNe) if  (TPa + TPb + TPc + TPd + TPe + FNa +FNb + FNc + FNd + FNe)> 0 else 0
    Micro_average_f = (metrics_ks['f1'] +metrics_ka['f1'] + metrics_es['f1']+ metrics_ea['f1'] +metrics_d['f1']) / 5

    Macro_average_r = (metrics_ks['recall'] + metrics_ka['recall'] + metrics_es['recall'] + metrics_ea['recall']+ metrics_d['recall'])/ 5
    Macro_average_p = (metrics_ks['precision'] + metrics_ka['precision'] + metrics_es['precision'] + metrics_ea['precision']+ metrics_d['precision'])/ 5
    Macro_average_f = (metrics_ks['f1'] +metrics_ka['f1'] + metrics_es['f1']+ metrics_ea['f1'] +metrics_d['f1']) / 5

    data = [{'Relation': 'knowledge_skills', 'F1_score': metrics_ks['f1'], 'precision': metrics_ks['precision'], 'recall': metrics_ks['recall']},
         {'Relation': 'knowledge_areas',  'F1_score': metrics_ka['f1'], 'precision': metrics_ka['precision'], 'recall': metrics_ka['recall']},
         {'Relation': 'experience_skills',  'F1_score': metrics_es['f1'], 'precision': metrics_es['precision'], 'recall': metrics_es['recall']},
        {'Relation': 'experience_areas',  'F1_score': metrics_ea['f1'], 'precision': metrics_ea['precision'], 'recall': metrics_ea['recall']},
         {'Relation': 'degree_in',  'F1_score': metrics_d['f1'], 'precision': metrics_d['precision'], 'recall': metrics_d['recall'] },
       {'Relation': 'All',  'F1_score': test_metrics['f1'], 'precision': test_metrics['precision'], 'recall': test_metrics['recall'] },
       {'Relation': 'Micro-average',  'F1_score': Micro_average_f, 'precision': Micro_average_p, 'recall': Micro_average_r },
       {'Relation': 'Macro-average',  'F1_score': Macro_average_f , 'precision': Macro_average_p, 'recall': Macro_average_r }]
  
    df_cf = pd.DataFrame.from_dict(data)
  
    df_cf.to_csv('./experiments/ex1/relation_test_confusion_metrics.csv')

    with open(params.data_dir / f'{mode}_triples.json', 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {    'prefix':"WebNLG",
                'target_text': [sample['text'] for sample in src],
                'input_text':  [" && ".join(p) for p in predictions_revised],
                'truth': [" && ".join(g) for g in ground_truths_revised]
                
            }
        )
        df.to_csv(params.ex_dir / f'{mode}_result.csv')
    logging.info('-done')