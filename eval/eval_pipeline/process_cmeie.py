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

with open('../../datas/ori/CMeIE-V2/53_schemas.json','r') as f:
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
    elif read_dir_name == 'c_to_ht':
        inp_prompt = '''当前你是一个资深的关系分类专家。
你的任务是基于给定文本和主客体实体对以及他们的类型，从给定关系列表中确定可能的关系。
给定的关系列表：{relation_list}。
任务输入的实体对格式是（主体||主体类型||客体||客体类型）。
任务的输出形式是（关系类型）。
给定文本：“{text}”
给定主客体实体对：“{head_tail}”
'''
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'subject_type':item[1],
                'object':item[2],
                'object_type':item[3]
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                sub = spo['subject']
                sub_type = spo['subject_type']
                obj = spo['object']
                obj_type = spo['object_type']
                ht_item_str = '（{}||{}||{}||{}）'.format(sub, sub_type, obj, obj_type)
                inp = inp_prompt.format(relation_list='，'.join(trip_types),text=text,head_tail=ht_item_str)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'subject_1':sub,
                    'subject_type_1':sub_type,
                    'object_1':obj,
                    'object_type_1':obj_type
                }
                new_datas.append(new_data)    
        return preds,new_datas,'htc_to_r'

    elif read_dir_name == 'c_to_r':
        inp_prompt = '''当前你是一个资深的信息提取的专家。
你的任务是从给定文本和关系类型，抽取所有可能的主客体实体对。先抽取可能存在的主体跨度，再基于抽取的主体跨度和给定文本继续抽取对应的客体跨度。并分别基于主/客体从给定实体列表中提取主/客体的实体类型。
给定的实体类型列表：{entity_list}。
任务的输出形式是（主体||主体类型||客体||客体类型）。
给定文本：“{text}”
给定关系类型：“{relation}”
'''
        re_grex = re.compile(r'（([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'predicate':item,
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                relation = spo['predicate']
                inp = inp_prompt.format(entity_list = '，'.join(ent_types),text=text,relation=relation)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'predicate_1':relation
                }
                new_datas.append(new_data)    
        return preds,new_datas,'rc_to_ht'

    elif read_dir_name == 'rc_to_ht':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'subject_type':item[1],
                'object':item[2],
                'object_type':item[3]
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''
    elif read_dir_name == 'htc_to_r':
        re_grex = re.compile(r'（([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'predicate':item
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''

def process_and_build_datas_notype(preds,read_dir_name,file_type):
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
    elif read_dir_name == 'c_to_ht':
        inp_prompt = '''当前你是一个资深的关系分类专家。
你的任务是基于给定文本和主客体实体对，从给定关系列表中确定可能的关系。
给定的关系列表：{relation_list}。
任务输入的实体对格式是（主体||客体）。
任务的输出形式是（关系类型）。
给定文本：“{text}”
给定主客体实体对：“{head_tail}”
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
                'object':item[1],
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                sub = spo['subject']
                obj = spo['object']
                ht_item_str = '（{}||{}）'.format(sub, obj)
                inp = inp_prompt.format(relation_list='，'.join(trip_types),text=text,head_tail=ht_item_str)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'subject_1':sub,
                    'object_1':obj
                }
                new_datas.append(new_data)    
        return preds,new_datas,'htc_to_r'

    elif read_dir_name == 'c_to_r':
        inp_prompt = '''当前你是一个资深的信息提取的专家。
你的任务是从给定文本和关系类型，抽取所有可能的主客体实体对。先抽取可能存在的主体跨度，再基于抽取的主体跨度和给定文本继续抽取对应的客体跨度。
给定的实体类型列表：{entity_list}。
任务的输出形式是（主体||客体）。
给定文本：“{text}”
给定关系类型：“{relation}”
'''
        re_grex = re.compile(r'（([^\|]*?)）')
        processed_preds = []
        new_datas = []
        for data_index,pred in enumerate(preds):
            response = pred['response']
            text = pred['text']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'predicate':item,
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                relation = spo['predicate']
                inp = inp_prompt.format(entity_list = '，'.join(ent_types),text=text,relation=relation)
                new_data = {
                    'instruction':inp,
                    'input':'',
                    'text':text,
                    'source_index':data_index,
                    'predicate_1':relation
                }
                new_datas.append(new_data)    
        return preds,new_datas,'rc_to_ht'

    elif read_dir_name == 'rc_to_ht':
        re_grex = re.compile(r'（([^\|]*?)\|\|([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'subject':item[0],
                'object':item[1]
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''
    elif read_dir_name == 'htc_to_r':
        re_grex = re.compile(r'（([^\|]*?)）')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'predicate':item
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''