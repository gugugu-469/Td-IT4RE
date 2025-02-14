seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



model_1="/home/wangjiacheng/llm_models/Meta-Llama-3.1-8B-Instruct"
version_model="ori"
version_type="with_type"
model_dtype="fp16"
template="llama3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
    --model_name_or_path_list ${model_1} \
    --version_type ${version_type} \
    --version_model ${version_model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --predict_nums -1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT/0_model_1_withtype/llama3_all_lora_07-24-01-39-28_export_model"
# version_model="v0"
# version_type="with_type"
# model_dtype="fp16"
# template="llama3"
# split_max_len=8192


# # 打印model和dataset的组合
# echo "Model: $model"
# echo "dataset_dir: $dataset_dir"
# CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
#     --model_name_or_path_list ${model_1} \
#     --version_type ${version_type} \
#     --version_model ${version_model} \
#     --template ${template} \
#     --dataset_dir ${dataset_dir} \
#     --n_shot 0 \
#     --n_avg 1 \
#     --predict_nums -1 \
#     --seed ${seed} \
#     --use_vllm \
#     --split_max_len ${split_max_len}

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/1_model_1_withtype/ade_llama3_all_lora_01-20-20-11-19"
version_model="v1"
version_type="with_type"
model_dtype="fp16"
template="llama3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
    --model_name_or_path_list ${model_1} \
    --version_type ${version_type} \
    --version_model ${version_model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --predict_nums -1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_1_withtype/ade_llama3_all_lora_01-20-23-29-36"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_2_withtype/ade_llama3_all_lora_01-21-01-25-48"
version_model="v2"
version_type="with_type"
model_dtype="fp16"
template="llama3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
    --model_name_or_path_list ${model_1},${model_2} \
    --version_type ${version_type} \
    --version_model ${version_model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --predict_nums -1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_1_withtype/ade_llama3_all_lora_01-27-08-36-04"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_2_withtype/ade_llama3_all_lora_01-27-09-21-39"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_3_withtype/ade_llama3_all_lora_01-27-10-05-09"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_4_withtype/ade_llama3_all_lora_01-21-02-13-07"
version_model="v3"
version_type="with_type"
model_dtype="fp16"
template="llama3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
    --model_name_or_path_list ${model_1},${model_2},${model_3},${model_4} \
    --version_type ${version_type} \
    --version_model ${version_model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --predict_nums -1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_withtype/ade_llama3_all_lora_01-21-03-49-26"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_withtype/ade_llama3_all_lora_01-21-06-51-04"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_withtype/ade_llama3_all_lora_01-21-03-49-26"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_withtype/ade_llama3_all_lora_01-21-06-51-04"
version_model="v4"
version_type="with_type"
model_dtype="fp16"
template="llama3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
    --model_name_or_path_list ${model_1},${model_2},${model_3},${model_4} \
    --version_type ${version_type} \
    --version_model ${version_model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --predict_nums -1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_withtype/ade_llama3_all_lora_01-21-09-09-12"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_withtype/ade_llama3_all_lora_01-21-11-28-09"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_withtype/ade_llama3_all_lora_01-21-09-09-12"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_withtype/ade_llama3_all_lora_01-21-11-28-09"
version_model="v5"
version_type="with_type"
model_dtype="fp16"
template="llama3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
    --model_name_or_path_list ${model_1},${model_2},${model_3},${model_4} \
    --version_type ${version_type} \
    --version_model ${version_model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --predict_nums -1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}





