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
from vllm.distributed.parallel_state import destroy_model_parallel

import gc
import ray

def get_time(fmt='%Y-%m-%d %H:%M:%S'):
    """
    获取当前时间，并增加8小时
    """
    # 获取当前时间
    ts = time.time()
    current_time = datetime.fromtimestamp(ts)

    # 增加8小时
    adjusted_time = current_time + timedelta(hours=8)

    # 格式化时间
    return adjusted_time.strftime(fmt)



def evaluate(
    model_name_or_path_list,
    version_model: Literal['ori','v0','v1','v2','v3','v4'],
    version_type: Literal['with_type','no_type'], 
    dataset_dir: Optional[str] = "../数据",
    model_dtype: Optional[str] = 'bf16',
    template: Optional[str] = 'chatglm3',
    n_shot: Optional[int] = 2,
    n_avg: Optional[int] = 1,
    seed: Optional[int] = 42,
    output_dir: Optional[str] = "./pred_output",
    predict_nums: Optional[int] = -1,
    use_vllm: Optional[Literal[False,True]] = False,
    split_max_len: Optional[int] = 8192,
):
    model_name_or_path_list = model_name_or_path_list.split(',')
    print(model_name_or_path_list)
    print(len(model_name_or_path_list))
    if 'CMeIE' in dataset_dir:
        from process_cmeie import process_and_build_datas_withtype,process_and_build_datas_notype
    else:
        from process_ade import process_and_build_datas_withtype,process_and_build_datas_notype

    if version_type == 'with_type':
        process_and_build_datas = process_and_build_datas_withtype
    else:
        process_and_build_datas = process_and_build_datas_notype
    transformers.set_seed(seed)
    random.seed(seed)
    out_time = get_time('%m-%d-%H-%M-%S')
    print('out_time:{}'.format(out_time))
    output_dataset_name = '{}|{}|{}|{}'.format(template,version_model,version_type,out_time)
    output_dir = os.path.join(output_dir,output_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir,'args.txt'),'w',encoding='utf-8') as f:
        f.write('out_time:{}\n'.format(out_time))
        f.write('template:{}\n'.format(template))
        f.write('model_name_or_path_list:{}\n'.format(model_name_or_path_list))
        f.write('version_type:{}\n'.format(version_type))
        f.write('version_model:{}\n'.format(version_model))
        f.write('seed:{}\n'.format(seed))


    if version_model == 'v1' or version_model == 'ori':
        # 一个模型
        assert len(model_name_or_path_list) == 1
    elif version_model == 'v0' :
        # 一个模型
        assert len(model_name_or_path_list) == 1
    elif version_model == 'v2':
        # 两个模型
        assert len(model_name_or_path_list) == 2
    elif version_model == 'v3':
        # 四个模型
        assert len(model_name_or_path_list) == 4
    elif version_model == 'v4':
        # 两个模型
        assert len(model_name_or_path_list) == 4
    elif version_model == 'v5':
        # 两个模型
        assert len(model_name_or_path_list) == 4
    
# c_to_ht, rc_to_ht, htc_to_r, c_to_hrt
    if version_type == 'no_type':
        data_path_extra = ''
    else:
        data_path_extra = '_withtype'

    if version_model == 'v0' or version_model == 'ori':
        dir_name_list= [['c_to_hrt']]
    elif version_model == 'v1':
        dir_name_list= [['c_to_hrt', 'c_to_ht', 'c_to_r', 'rc_to_ht', 'htc_to_r']]
    elif version_model == 'v2':
        dir_name_list= [['c_to_hrt','c_to_ht','c_to_r'],['c_to_hrt','rc_to_ht', 'htc_to_r']]
    elif version_model == 'v3':
        dir_name_list= [['c_to_hrt','c_to_ht'],['c_to_hrt','c_to_r'],['c_to_hrt','rc_to_ht'],['c_to_hrt','htc_to_r']]
    elif version_model == 'v4':
        dir_name_list= [['c_to_hrt','c_to_ht'],['c_to_hrt','c_to_r'],['rc_to_ht'],['htc_to_r']]
    elif version_model == 'v5':
        dir_name_list= [['c_to_hrt','c_to_ht'],['c_to_hrt','c_to_r'],['htc_to_r'],['rc_to_ht']]

    files = []
    outs = []
    for sub_dir_list in dir_name_list:
        tmp_out_list = []
        tmp_file_list = []
        for dir_name in sub_dir_list:
            read_dir = os.path.join(dataset_dir,'{}{}'.format(dir_name,data_path_extra))
            if 'CMeIE' in dataset_dir:
                tmp_file_list.append((dir_name,'dev',os.path.join(read_dir,'dev.json')))
            else:
                tmp_file_list.append((dir_name,'test',os.path.join(read_dir,'test.json')))
            # tmp_file_list.append((dir_name,'test',os.path.join(read_dir,'test.json')))
            out_dir = os.path.join(output_dir,'{}{}'.format(dir_name,data_path_extra))
            os.makedirs(out_dir,exist_ok=True)
            tmp_out_list.append(out_dir)
            # tmp_out_list.append(out_dir)
        files.append(tmp_file_list)
        outs.append(tmp_out_list)

    # 加载模型
    if model_dtype == 'fp16':
        use_type = 'float16'
    elif model_dtype == 'bf16':
        use_type = 'bfloat16'
    else:
        use_type = model_dtype


    processed_datas = {}
    for model_index,model_name_or_path in enumerate(model_name_or_path_list):
        if model_index != 0:
            destroy_model_parallel()
            del model.llm_engine.model_executor.driver_worker
            del model

            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()

            print(f"cuda memory: {torch.cuda.memory_allocated() // 1024 // 1024}MB")

            print("service stopped")

        model = LLM(model=model_name_or_path,tokenizer_mode='auto', trust_remote_code = True,dtype=use_type, max_model_len = split_max_len, gpu_memory_utilization=0.85)
        sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens = split_max_len)
        tokenizer = model.get_tokenizer()

        
        start_time = time.time()
        for (read_dir_name,file_type,file),out_dir in zip(files[model_index],outs[model_index]):
            # 拿到test集，并且是jsonl文件
            print('now file:{}'.format(file))

            if 'rc_to_ht' in read_dir_name or 'htc_to_r' in read_dir_name:
                # 需要后处理的数据
                test_samples = processed_datas['{}_{}'.format(read_dir_name,file_type)]
            else:
                test_samples = []
                with jsonlines.open(file,'r') as f:
                    for data in f:
                        test_samples.append(data)

            if predict_nums == -1:
                predict_datas = test_samples
            elif predict_nums <= 0:
                raise ValueError("predict nums设定错误，应为-1或大于0")
            else:
                predict_datas = test_samples[:predict_nums]
            
            preds = []
            prompts = []
            # 可以预测的下标
            corrects = {}
            # 不可以预测的下标
            errors = []

            processed_prompt_list = []
            for s_index,sample in enumerate(predict_datas):
                messages = [
                    {"role": "user", "content": sample['instruction']}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                corrects[s_index] = len(corrects.keys())
                processed_prompt_list.append(text)
            file_start = time.time()
            outputs = model.generate(processed_prompt_list, sampling_params)
            file_end = time.time()
            # 输出
            correct_indexes = corrects.keys()
            print('得到输出数据')
            # 赋值pred即可
            for s_index,sample in tqdm(enumerate(predict_datas)):
                output_index = corrects[s_index]
                response = outputs[output_index].outputs[0].text
                sample['response'] = response
                preds.append(sample)
            # 处理一下，构造processed_datas
            p_datas, new_datas, new_name = process_and_build_datas(preds,read_dir_name,file_type)
            if new_name != '':

                processed_datas['{}_{}'.format(new_name,file_type)] = new_datas
                with jsonlines.open(os.path.join(out_dir,'{}_new_datas.json'.format(file_type)),'w') as f:
                    for data in new_datas:
                        f.write(data)
            print('out_dir:{}'.format(out_dir))
            print('file_type:{}'.format(file_type))
            with jsonlines.open(os.path.join(out_dir,'{}.json'.format(file_type)),'w') as f:
                for data in p_datas:
                    f.write(data)
            with jsonlines.open(os.path.join(out_dir,'{}.json'.format('cost_time')),'w') as f:
                f.write({
                    'cost_time': file_end-file_start
                })

if __name__ == "__main__":
    fire.Fire(evaluate)
