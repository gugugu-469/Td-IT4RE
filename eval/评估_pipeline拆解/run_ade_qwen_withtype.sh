seed=2024
gpu=1
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"


model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/1_model_1_withtype/ade_qwen2.5_all_lora_02-04-09-06-42"
version_model="v1"
version_type="with_type"
model_dtype="bf16"
template="qwen"
split_max_len=2000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/2_model_1_withtype/ade_qwen2.5_all_lora_02-04-11-13-26"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/2_model_2_withtype/ade_qwen2.5_all_lora_02-04-12-20-29"
version_model="v2"
version_type="with_type"
model_dtype="bf16"
template="qwen"
split_max_len=2000


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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_1_withtype/ade_qwen2.5_all_lora_02-04-14-26-18"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_2_withtype/ade_qwen2.5_all_lora_02-04-15-38-18"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_3_withtype/ade_qwen2.5_all_lora_02-04-16-48-08"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/3_model_4_withtype/ade_qwen2.5_all_lora_02-04-18-01-08"
version_model="v3"
version_type="with_type"
model_dtype="bf16"
template="qwen"
split_max_len=2000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_1_withtype/ade_qwen2.5_all_lora_02-04-19-31-20"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_2_withtype/ade_qwen2.5_all_lora_02-04-21-18-06"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_1_withtype/ade_qwen2.5_all_lora_02-04-19-31-20"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/4_model_2_withtype/ade_qwen2.5_all_lora_02-04-21-18-06"
version_model="v4"
version_type="with_type"
model_dtype="bf16"
template="qwen"
split_max_len=2000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_1_withtype/ade_qwen2.5_all_lora_02-04-23-17-41"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_2_withtype/ade_qwen2.5_all_lora_02-05-01-18-45"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_1_withtype/ade_qwen2.5_all_lora_02-04-23-17-41"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models_export/RE_SFT_0104_pipeline/5_model_2_withtype/ade_qwen2.5_all_lora_02-05-01-18-45"
version_model="v5"
version_type="with_type"
model_dtype="bf16"
template="qwen"
split_max_len=2000


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





