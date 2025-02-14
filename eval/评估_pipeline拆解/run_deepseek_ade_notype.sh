seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"




# model_1="/home/wangjiacheng/mnt_mine/llm_models/deepseek-llm-7b-chat"
# version_model="ori"
# version_type="no_type"
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

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_notype/ade_deepseek_all_lora_01-03-18-59-26/checkpoint-416"
# model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_notype/ade_deepseek_all_lora_01-03-18-59-26/checkpoint-416"
# model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_notype/ade_deepseek_all_lora_01-03-18-59-26/checkpoint-416"
# version_model="v0"
# version_type="no_type"
# model_dtype="bf16"
# template="deepseek"
# split_max_len=4096


# # 打印model和dataset的组合
# echo "Model: $model"
# echo "dataset_dir: $dataset_dir"
# CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
#     --model_name_or_path_list ${model_1},${model_2},${model_3} \
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/1_model_1_notype/ade_deepseek_all_lora_01-17-02-57-06"
version_model="v1"
version_type="no_type"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_1_notype/ade_deepseek_all_lora_01-17-06-07-28"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_2_notype/ade_deepseek_all_lora_01-17-07-58-56"
version_model="v2"
version_type="no_type"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_1_notype/ade_deepseek_all_lora_01-17-09-59-33"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_2_notype/ade_deepseek_all_lora_01-17-11-00-43"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_3_notype/ade_deepseek_all_lora_01-17-12-01-00"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_4_notype/ade_deepseek_all_lora_01-17-13-03-32"
version_model="v3"
version_type="no_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/ade_deepseek_all_lora_01-17-14-24-08"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/ade_deepseek_all_lora_01-17-15-59-01"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/ade_deepseek_all_lora_01-17-14-24-08"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/ade_deepseek_all_lora_01-17-15-59-01"
version_model="v4"
version_type="no_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/ade_deepseek_all_lora_01-17-17-50-25"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/ade_deepseek_all_lora_01-17-19-37-09"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/ade_deepseek_all_lora_01-17-17-50-25"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/ade_deepseek_all_lora_01-17-19-37-09"
version_model="v5"
version_type="no_type"
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





