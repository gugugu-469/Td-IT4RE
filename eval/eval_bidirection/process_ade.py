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
with jsonlines.open('../../datas/ori/ADE/65_schemas.json','r') as f:
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
    elif read_dir_name == 'c_to_hr':
        inp_prompt = '''Currently, you are a senior expert in information extraction. 
Your task is to first extract possible objects from the given text, the subject, and the subject type, and then identify possible entity types of these objects from the given entity type list. The given entity type list is: {entity_list}.
Next, based on the subject-object entity pairs and their corresponding entity types, identify possible relations from the given relation list. The given relation list is: {relation_list}.
The output format for the task is (object||entity type||relation).
The given text:"{text}; Subject: {subject}; Subject entity type: {subject_type}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
                inp = inp_prompt.format(relation_list='["' +'","'.join(trip_types) + '"]',entity_list = '["' +'","'.join(ent_types)+ '"]',text=text,subject=sub,subject_type = sub_type)
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
        inp_prompt = '''Currently, you are a senior expert in information extraction. 
Your task is to first extract possible subjects from the given text, the object, and the object type, and then identify possible entity types of these subjects from the given entity type list. The given entity type list is: {entity_list}.
Next, based on the subject-object entity pairs and their corresponding entity types, identify possible relations from the given relation list. The given relation list is: {relation_list}.
The output format for the task is (subject||entity type||relation).
The given text:"{text}; Object: {object}; Object entity type: {object_type}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
                inp = inp_prompt.format(relation_list='["' +'","'.join(trip_types) + '"]',entity_list = '["' +'","'.join(ent_types)+ '"]',text=text,object=obj,object_type = obj_type)
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\|\|([^\|]*?)\)')
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
        inp_prompt = '''Currently, you are a senior expert in information extraction. 
Your task is to first extract possible objects from the given text and the subject, and then, based on the subject-object entity pairs, identify possible relations from the given relation list. The given relation list is: {relation_list}.
The output format for the task is (object||relation).
The given text:"Text: {text}; Subject: {subject}"
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
                'predicate':item[1],
            } for item in find_tuples]
            pred['output'] = spo_list
            for spo in spo_list:
                sub = spo['subject']
                rel = spo['predicate']
                inp = inp_prompt.format(relation_list='["' +'","'.join(trip_types) + '"]',text=text,subject=sub)
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
        inp_prompt = '''Currently, you are a senior expert in information extraction. 
Your task is to first extract possible subjects from the given text and the object, and then, based on the subject-object entity pairs, identify possible relations from the given relation list. The given relation list is: {relation_list}.
The output format for the task is (subject||relation).
The given text:"Text: {text}; Object: {object}"
'''
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\)')
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
                inp = inp_prompt.format(relation_list='["' +'","'.join(trip_types) + '"]',text=text,object=obj)
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\)')
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
        re_grex = re.compile(r'\(([^\|]*?)\|\|([^\|]*?)\)')
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