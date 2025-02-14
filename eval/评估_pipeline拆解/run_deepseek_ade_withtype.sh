seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"




model_1="/home/wangjiacheng/mnt_mine/llm_models/deepseek-llm-7b-chat"
version_model="ori"
version_type="with_type"
model_dtype="bf16"
template="deepseek"
split_max_len=4096


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

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_export/0_model_1_withtype/ade_deepseek_all_lora_01-04-15-26-14"
# version_model="v0"
# version_type="with_type"
# model_dtype="bf16"
# template="deepseek"
# split_max_len=4096


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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/1_model_1_withtype/ade_deepseek_all_lora_01-17-21-03-07"
version_model="v1"
version_type="with_type"
model_dtype="bf16"
template="deepseek"
split_max_len=4096


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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_1_withtype/ade_deepseek_all_lora_01-17-23-43-52"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_2_withtype/ade_deepseek_all_lora_01-18-01-10-01"
version_model="v2"
version_type="with_type"
model_dtype="bf16"
template="deepseek"
split_max_len=4096


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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_1_withtype/ade_deepseek_all_lora_01-18-03-02-12"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_2_withtype/ade_deepseek_all_lora_01-18-04-02-46"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_3_withtype/ade_deepseek_all_lora_01-18-05-06-26"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_4_withtype/ade_deepseek_all_lora_01-18-06-09-20"
version_model="v3"
version_type="with_type"
model_dtype="bf16"
template="deepseek"
split_max_len=4096


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_withtype/ade_deepseek_all_lora_01-18-07-30-27"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_withtype/ade_deepseek_all_lora_01-18-09-10-11"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_withtype/ade_deepseek_all_lora_01-18-07-30-27"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_withtype/ade_deepseek_all_lora_01-18-09-10-11"
version_model="v4"
version_type="with_type"
model_dtype="bf16"
template="deepseek"
split_max_len=4096


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_withtype/ade_deepseek_all_lora_01-18-11-07-08"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_withtype/ade_deepseek_all_lora_01-18-13-10-43"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_withtype/ade_deepseek_all_lora_01-18-11-07-08"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_withtype/ade_deepseek_all_lora_01-18-13-10-43"
version_model="v5"
version_type="with_type"
model_dtype="bf16"
template="deepseek"
split_max_len=4096


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





