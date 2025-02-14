seed=2024
gpu=1
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"

model_1="/home/wangjiacheng/llm_models/Qwen2___5-7B-Instruct"
version_model="ori"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/0_model_1_notype/ade_qwen2.5_all_lora_02-03-16-37-22"
version_model="v0"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/1_model_1_notype/ade_qwen2.5_all_lora_02-03-17-15-01"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/2_model_1_notype/ade_qwen2.5_all_lora_02-03-20-24-33"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/2_model_2_notype/ade_qwen2.5_all_lora_02-03-22-04-59"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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


model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_1_notype/ade_qwen2.5_all_lora_02-03-23-26-59"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_2_notype/ade_qwen2.5_all_lora_02-04-00-35-53"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_3_notype/ade_qwen2.5_all_lora_02-04-01-46-57"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_4_notype/ade_qwen2.5_all_lora_02-04-03-00-43"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_1_notype/ade_qwen2.5_all_lora_02-04-04-22-11"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_2_notype/ade_qwen2.5_all_lora_02-04-06-08-37"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_1_notype/ade_qwen2.5_all_lora_02-04-04-22-11"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_2_notype/ade_qwen2.5_all_lora_02-04-06-08-37"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_1_notype/ade_qwen2.5_all_lora_02-04-07-58-54"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_2_notype/ade_qwen2.5_all_lora_02-04-09-16-10"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_1_notype/ade_qwen2.5_all_lora_02-04-07-58-54"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_2_notype/ade_qwen2.5_all_lora_02-04-09-16-10"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=4096


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_predict_2.py \
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





