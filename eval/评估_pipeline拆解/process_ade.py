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

trip_types_list = []
with jsonlines.open('/home/wangjiacheng/面向RE任务微调LLM/datas/ori/ADE/65_schemas.json','r') as f:
    for data in f:
        trip_types_list.append(data)
print(trip_types_list)
trip_types = sorted(set([data['predicate'] for data in trip_types_list]))
ent_type_list = []
for data in trip_types_list:
    ent_type_list.append(data['subject_type'])
    ent_type_list.append(data['object_type'])
ent_types = sorted(set(ent_type_list),reverse=True)

def process_and_build_datas_withtype(preds,read_dir_name,file_type):
    if read_dir_name == 'c_to_hrt':
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
        inp_prompt = '''Currently, you are a senior expert in relation classification.
Your task is to determine possible relations from the given relation list based on the given text, subject-object entity pairs, and their types.
The given relation list is: {relation_list}.
The input format for entity pairs is (subject||subject type||object||object type).
The output format for the task is (relation type).
The given text: "{text}"
The given subject-object entity pair: "{head_tail}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
                ht_item_str = '({}||{}||{}||{})'.format(sub, sub_type, obj, obj_type)
                inp = inp_prompt.format(relation_list='["' +'","'.join(trip_types) + '"]',text=text,head_tail=ht_item_str)
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
        inp_prompt = '''Currently, you are a senior expert in information extraction.
Your task is to extract all possible subject-object entity pairs from the given text and relation type. First, extract possible subject spans, and based on the extracted subject spans and the given text, continue to extract the corresponding object spans. Then, identify the entity types of the subject and object from the given entity type list.
The given entity type list is: {entity_list}.
The output format for the task is (subject||subject type||object||object type).
The given text: "{text}"
The given relation type: "{relation}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\)')
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
                inp = inp_prompt.format(entity_list = '["' +'","'.join(ent_types)+ '"]',text=text,relation=relation)
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
        re_grex = re.compile(r'\(([^\|]*?)\)')
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
        inp_prompt = '''Currently, you are a senior expert in relation classification.
Your task is to determine possible relations from the given relation list based on the given text and subject-object entity pairs.
The given relation list is: {relation_list}.
The input format for entity pairs is (subject||object).
The output format for the task is (relation type).
The given text: "{text}"
The given subject-object entity pair: "{head_tail}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\)')
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
                ht_item_str = '({}||{})'.format(sub, obj)
                inp = inp_prompt.format(relation_list='["' +'","'.join(trip_types) + '"]',text=text,head_tail=ht_item_str)
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
        inp_prompt = '''Currently, you are a senior expert in information extraction.
Your task is to extract all possible subject-object entity pairs from the given text and relation type. You need to extract possible subject spans, and based on the extracted subject spans and the given text, continue to extract the corresponding object spans.
The given entity type list is: {entity_list}.
The output format for the task is (subject||object).
The given text: "{text}"
The given relation type: "{relation}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\)')
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
                inp = inp_prompt.format(entity_list = '["' +'","'.join(ent_types)+ '"]',text=text,relation=relation)
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\)')
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
        re_grex = re.compile(r'\(([^\|]*?)\)')
        processed_preds = []
        for pred in preds:
            response = pred['response']
            find_tuples = re_grex.findall(response)
            spo_list = [{
                'predicate':item
            } for item in find_tuples]
            pred['output'] = spo_list
        return preds,[],''