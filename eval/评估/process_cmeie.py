import os
from misc import *
import torch
import fire
from transformers import GenerationConfig, TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, AutoModel
import transformers
from types import MethodType
import json
import jsonlines
from collections import defaultdict
import random
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from vllm import LLM, SamplingParams
import re

with open('/home/wangjiacheng/面向RE任务微调LLM/datas/ori/CMeIE-V2/53_schemas.json','r') as f:
    trip_types_list = json.load(f)
trip_types = sorted(set([data['predicate'] for data in trip_types_list]))
ent_type_list = []
for data in trip_types_list:
    ent_type_list.append(data['subject_type'])
    ent_type_list.append(data['object_type'])
ent_types = sorted(set(ent_type_list),reverse=True)

def process_and_build_datas_withtype(preds,read_dir_name,file_type):
    if read_dir_name == 'c_to_hrt':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'subject_type':item[1],
                'predicate':item[2],
                'object':item[3],
                'object_type':item[4]
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''
    elif read_dir_name == 'c_to_hr':
        inp_prompt = '''当前你是一个资深的信息提取的专家。
你的任务是从给定文本、主体和主体类型，先抽取可能存在的客体，并基于客体从给定实体列表中提取可能的实体类型。给定的实体类型列表：{entity_list}。
然后基于主客实体对及其对应的实体类型从给定关系列表中提取可能的关系。给定的关系列表{relation_list}。
任务的输出形式是（客体||实体类型||关系类型）。
给定文本：“文本：{text}；主体：{subject}；主体类型：{subject_type}”
'''
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'subject_type':item[1],
                'predicate':item[2],
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                sub = spo['subject']
                rel = spo['predicate']
                sub_type = spo['subject_type']
                inp = inp_prompt.format(relation_list='，'.join(trip_types),entity_list = '，'.join(ent_types),text=text,subject=sub,subject_type = sub_type)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'subject_1':sub,
                    'subject_type_1':sub_type,
                    'predicate_1':rel
                }
                new_datas.append(new_data)    
        return preds,new_datas,'sc_to_tr'

    elif read_dir_name == 'c_to_tr':
        inp_prompt = '''当前你是一个资深的信息提取的专家。
你的任务是从给定文本、客体和客体类型，先抽取可能存在的主体，并基于主体从给定实体列表中提取可能的实体类型。给定的实体类型列表：{entity_list}。
然后基于主客实体对及其对应的实体类型从给定关系列表中提取可能的关系。给定的关系列表{relation_list}。
任务的输出形式是（主体||实体类型||关系类型）。
给定文本：“文本：{text}；客体：{object}；客体类型：{object_type}”
'''
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'object':item[0],
                'object_type':item[1],
                'predicate':item[2],
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                obj = spo['object']
                rel = spo['predicate']
                obj_type = spo['object_type']
                inp = inp_prompt.format(relation_list='，'.join(trip_types),entity_list = '，'.join(ent_types),text=text,object=obj,object_type = obj_type)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'object_1':obj,
                    'object_type_1':obj_type,
                    'predicate_1':rel
                }
                new_datas.append(new_data)    
        return preds,new_datas,'sc_to_hr'

    elif read_dir_name == 'sc_to_hr':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'subject_type':item[1],
                'predicate':item[2]
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''
    elif read_dir_name == 'sc_to_tr':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'object':item[0],
                'object_type':item[1],
                'predicate':item[2],
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''

def process_and_build_datas_notype(preds,read_dir_name,file_type):
    if read_dir_name == 'c_to_hrt':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'predicate':item[1],
                'object':item[2],
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''
    elif read_dir_name == 'c_to_hr':
        inp_prompt = '''当前你是一个资深的信息提取的专家。
你的任务是从给定文本和主体，先抽取可能存在的客体，再基于主客实体对从给定关系列表中提取可能的关系。给定的关系列表：{type_list}。
任务的输出形式是（客体||关系类型）。
给定文本：“{text}；主体：{subject}”
'''
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'predicate':item[1],
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                sub = spo['subject']
                rel = spo['predicate']
                inp = inp_prompt.format(type_list='，'.join(trip_types),text=text,subject=sub)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'subject_1':sub,
                    'predicate_1':rel
                }
                new_datas.append(new_data)    
        return preds,new_datas,'sc_to_tr'

    elif read_dir_name == 'c_to_tr':
        inp_prompt = '''当前你是一个资深的信息提取的专家。
你的任务是从给定文本和客体，先抽取可能存在的主体，再基于主客实体对从给定关系列表中提取可能的关系。给定的关系列表：{type_list}。
任务的输出形式是（主体||关系类型）。
给定文本：“{text}；客体：{object}”
'''
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'object':item[0],
                'predicate':item[1],
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                obj = spo['object']
                rel = spo['predicate']
                inp = inp_prompt.format(type_list='，'.join(trip_types),text=text,object=obj)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'object_1':obj,
                    'predicate_1':rel
                }
                new_datas.append(new_data)    
        return preds,new_datas,'sc_to_hr'

    elif read_dir_name == 'sc_to_hr':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'predicate':item[1]
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''
    elif read_dir_name == 'sc_to_tr':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'object':item[0],
                'predicate':item[1],
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''