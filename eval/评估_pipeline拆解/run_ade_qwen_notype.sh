seed=2024
gpu=1
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/1_model_1_notype/ade_qwen2.5_all_lora_02-03-16-37-34"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="qwen"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/2_model_1_notype/ade_qwen2.5_all_lora_02-03-18-42-05"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/2_model_2_notype/ade_qwen2.5_all_lora_02-03-19-47-37"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="qwen"
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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_1_notype/ade_qwen2.5_all_lora_02-03-21-05-32"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_2_notype/ade_qwen2.5_all_lora_02-03-22-17-20"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_3_notype/ade_qwen2.5_all_lora_02-03-23-29-33"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_4_notype/ade_qwen2.5_all_lora_02-04-00-39-25"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="qwen"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_1_notype/ade_qwen2.5_all_lora_02-04-02-07-13"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_2_notype/ade_qwen2.5_all_lora_02-04-03-51-46"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_1_notype/ade_qwen2.5_all_lora_02-04-02-07-13"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_2_notype/ade_qwen2.5_all_lora_02-04-03-51-46"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="qwen"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_1_notype/ade_qwen2.5_all_lora_02-04-05-53-51"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_2_notype/ade_qwen2.5_all_lora_02-04-07-56-09"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_1_notype/ade_qwen2.5_all_lora_02-04-05-53-51"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_2_notype/ade_qwen2.5_all_lora_02-04-07-56-09"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="qwen"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/1_model_1_notype/ade_qwen2.5_all_lora_02-03-16-37-34"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=8192


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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/2_model_1_notype/ade_qwen2.5_all_lora_02-03-18-42-05"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/2_model_2_notype/ade_qwen2.5_all_lora_02-03-19-47-37"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=8192


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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_1_notype/ade_qwen2.5_all_lora_02-03-21-05-32"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_2_notype/ade_qwen2.5_all_lora_02-03-22-17-20"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_3_notype/ade_qwen2.5_all_lora_02-03-23-29-33"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_4_notype/ade_qwen2.5_all_lora_02-04-00-39-25"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=8192


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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_1_notype/ade_qwen2.5_all_lora_02-04-02-07-13"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_2_notype/ade_qwen2.5_all_lora_02-04-03-51-46"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_1_notype/ade_qwen2.5_all_lora_02-04-02-07-13"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_2_notype/ade_qwen2.5_all_lora_02-04-03-51-46"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=8192


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_1_notype/ade_qwen2.5_all_lora_02-04-05-53-51"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_2_notype/ade_qwen2.5_all_lora_02-04-07-56-09"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_1_notype/ade_qwen2.5_all_lora_02-04-05-53-51"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_2_notype/ade_qwen2.5_all_lora_02-04-07-56-09"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="qwen"
split_max_len=8192


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


