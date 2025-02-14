seed=2024
gpu=1
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"

model_1="/home/wangjiacheng/llm_models/Qwen2___5-7B-Instruct"
version_model="ori"
version_type="with_type"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/0_model_1_withtype/ade_qwen2.5_all_lora_02-04-10-24-58"
version_model="v0"
version_type="with_type"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/1_model_1_withtype/ade_qwen2.5_all_lora_02-04-10-49-49"
version_model="v1"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/2_model_1_withtype/ade_qwen2.5_all_lora_02-04-12-55-04"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/2_model_2_withtype/ade_qwen2.5_all_lora_02-04-14-39-48"
version_model="v2"
version_type="with_type"
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


model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_1_withtype/ade_qwen2.5_all_lora_02-04-16-41-19"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_2_withtype/ade_qwen2.5_all_lora_02-04-17-30-26"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_3_withtype/ade_qwen2.5_all_lora_02-04-18-40-47"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/3_model_4_withtype/ade_qwen2.5_all_lora_02-04-19-34-23"
version_model="v3"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_1_withtype/ade_qwen2.5_all_lora_02-04-20-54-51"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_2_withtype/ade_qwen2.5_all_lora_02-04-22-10-47"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_1_withtype/ade_qwen2.5_all_lora_02-04-20-54-51"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/4_model_2_withtype/ade_qwen2.5_all_lora_02-04-22-10-47"
version_model="v4"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_1_withtype/ade_qwen2.5_all_lora_02-05-00-06-31"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_2_withtype/ade_qwen2.5_all_lora_02-05-02-01-16"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_1_withtype/ade_qwen2.5_all_lora_02-05-00-06-31"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104/5_model_2_withtype/ade_qwen2.5_all_lora_02-05-02-01-16"
version_model="v5"
version_type="with_type"
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





