seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/CMeIE-V2

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"








# model_1="/home/wangjiacheng/llm_models/glm-4-9b-chat"
# version_model="ori"
# version_type="no_type"
# model_dtype="bf16"
# template="glm4"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_notype/CMeIE_glm4_all_lora_01-07-20-21-33"
version_model="v0"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_notype/CMeIE_glm4_all_lora_01-08-00-55-12"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_notype/CMeIE_glm4_all_lora_01-09-04-22-48"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_notype/CMeIE_glm4_all_lora_01-09-17-49-24"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_notype/CMeIE_glm4_all_lora_01-07-14-08-24"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_notype/CMeIE_glm4_all_lora_01-10-22-20-02"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_notype/CMeIE_glm4_all_lora_01-11-06-46-52"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_notype/CMeIE_glm4_all_lora_01-12-05-01-33"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_notype/CMeIE_glm4_all_lora_01-12-15-56-44"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_notype/CMeIE_glm4_all_lora_01-13-16-35-55"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_notype/CMeIE_glm4_all_lora_01-12-15-56-44"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_notype/CMeIE_glm4_all_lora_01-13-16-35-55"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_notype/CMeIE_glm4_all_lora_01-14-05-52-34"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_notype/CMeIE_glm4_all_lora_01-14-18-46-05"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_notype/CMeIE_glm4_all_lora_01-14-05-52-34"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_notype/CMeIE_glm4_all_lora_01-14-18-46-05"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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






model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_notype/CMeIE_glm4_all_lora_01-07-20-21-33"
version_model="v0"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_notype/CMeIE_glm4_all_lora_01-08-00-55-12"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_notype/CMeIE_glm4_all_lora_01-09-04-22-48"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_notype/CMeIE_glm4_all_lora_01-09-17-49-24"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_notype/CMeIE_glm4_all_lora_01-07-14-08-24"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_notype/CMeIE_glm4_all_lora_01-10-22-20-02"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_notype/CMeIE_glm4_all_lora_01-11-06-46-52"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_notype/CMeIE_glm4_all_lora_01-12-05-01-33"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_notype/CMeIE_glm4_all_lora_01-12-15-56-44"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_notype/CMeIE_glm4_all_lora_01-13-16-35-55"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_notype/CMeIE_glm4_all_lora_01-12-15-56-44"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_notype/CMeIE_glm4_all_lora_01-13-16-35-55"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_notype/CMeIE_glm4_all_lora_01-14-05-52-34"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_notype/CMeIE_glm4_all_lora_01-14-18-46-05"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_notype/CMeIE_glm4_all_lora_01-14-05-52-34"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_notype/CMeIE_glm4_all_lora_01-14-18-46-05"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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





