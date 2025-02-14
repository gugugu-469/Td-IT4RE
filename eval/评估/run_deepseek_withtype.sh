seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"

# model_1="/home/wangjiacheng/llm_models/deepseek-llm-7b-chat"
# version_model="ori"
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

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_withtype/ade_deepseek_all_lora_01-04-15-26-14"
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

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_withtype/ade_deepseek_all_lora_01-04-17-59-59"
# version_model="v1"
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




# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_withtype/ade_deepseek_all_lora_01-04-20-53-20"
# model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_withtype/ade_deepseek_all_lora_01-04-22-29-32"
# version_model="v2"
# version_type="with_type"
# model_dtype="bf16"
# template="deepseek"
# split_max_len=4096


# # 打印model和dataset的组合
# echo "Model: $model"
# echo "dataset_dir: $dataset_dir"
# CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
#     --model_name_or_path_list ${model_1},${model_2} \
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













model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_withtype/ade_deepseek_all_lora_01-05-00-21-31"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_withtype/ade_deepseek_all_lora_01-05-01-25-58"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_withtype/ade_deepseek_all_lora_01-05-02-30-51"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_withtype/ade_deepseek_all_lora_01-05-03-42-22"
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




# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/ade_deepseek_all_lora_01-05-04-59-31"
# model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/ade_deepseek_all_lora_01-05-06-41-49"
# model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/ade_deepseek_all_lora_01-05-04-59-31"
# model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/ade_deepseek_all_lora_01-05-06-41-49"
# version_model="v4"
# version_type="with_type"
# model_dtype="bf16"
# template="deepseek"
# split_max_len=4096


# # 打印model和dataset的组合
# echo "Model: $model"
# echo "dataset_dir: $dataset_dir"
# CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
#     --model_name_or_path_list ${model_1},${model_2},${model_3},${model_4} \
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




# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/ade_deepseek_all_lora_01-05-08-26-17"
# model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/ade_deepseek_all_lora_01-05-10-10-20"
# model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/ade_deepseek_all_lora_01-05-08-26-17"
# model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/ade_deepseek_all_lora_01-05-10-10-20"
# version_model="v5"
# version_type="with_type"
# model_dtype="bf16"
# template="deepseek"
# split_max_len=4096


# # 打印model和dataset的组合
# echo "Model: $model"
# echo "dataset_dir: $dataset_dir"
# CUDA_VISIBLE_DEVICES=${gpu} python eval_predict.py \
#     --model_name_or_path_list ${model_1},${model_2},${model_3},${model_4} \
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





