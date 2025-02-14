seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/CMeIE-V2

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



# model_1="/home/wangjiacheng/llm_models/Qwen2___5-7B-Instruct"
# version_model="ori"
# version_type="with_type"
# model_dtype="bf16"
# template="qwen"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_withtype/CMeIE_qwen2.5_all_lora_01-03-14-00-59"
version_model="v0"
version_type="with_type"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_withtype/CMeIE_qwen2.5_all_lora_01-04-11-29-32"
version_model="v1"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_withtype/CMeIE_qwen2.5_all_lora_01-05-09-04-47"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_withtype/CMeIE_qwen2.5_all_lora_01-05-16-36-41"
version_model="v2"
version_type="with_type"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_withtype/CMeIE_qwen2.5_all_lora_01-06-09-33-34"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_withtype/CMeIE_qwen2.5_all_lora_01-06-14-40-03"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_withtype/CMeIE_qwen2.5_all_lora_01-06-19-55-08"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_withtype/CMeIE_qwen2.5_all_lora_01-07-08-58-35"
version_model="v3"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_qwen2.5_all_lora_01-08-12-59-15"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-04-27-55"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_qwen2.5_all_lora_01-08-12-59-15"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-04-27-55"
version_model="v4"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_qwen2.5_all_lora_01-09-13-43-52"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-22-49-23"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_qwen2.5_all_lora_01-09-13-43-52"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-22-49-23"
version_model="v5"
version_type="with_type"
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






model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_withtype/CMeIE_qwen2.5_all_lora_01-03-14-00-59"
version_model="v0"
version_type="with_type"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_withtype/CMeIE_qwen2.5_all_lora_01-04-11-29-32"
version_model="v1"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_withtype/CMeIE_qwen2.5_all_lora_01-05-09-04-47"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_withtype/CMeIE_qwen2.5_all_lora_01-05-16-36-41"
version_model="v2"
version_type="with_type"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_withtype/CMeIE_qwen2.5_all_lora_01-06-09-33-34"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_withtype/CMeIE_qwen2.5_all_lora_01-06-14-40-03"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_withtype/CMeIE_qwen2.5_all_lora_01-06-19-55-08"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_withtype/CMeIE_qwen2.5_all_lora_01-07-08-58-35"
version_model="v3"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_qwen2.5_all_lora_01-08-12-59-15"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-04-27-55"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_qwen2.5_all_lora_01-08-12-59-15"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-04-27-55"
version_model="v4"
version_type="with_type"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_qwen2.5_all_lora_01-09-13-43-52"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-22-49-23"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_qwen2.5_all_lora_01-09-13-43-52"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_qwen2.5_all_lora_01-09-22-49-23"
version_model="v5"
version_type="with_type"
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





