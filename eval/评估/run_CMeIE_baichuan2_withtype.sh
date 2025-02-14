seed=2024
gpu=6
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/CMeIE-V2

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



# model_1="/home/wangjiacheng/llm_models/Baichuan2-7B-Chat"
# version_model="ori"
# version_type="with_type"
# model_dtype="bf16"
# template="baichuan2"
# split_max_len=4000


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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_withtype/CMeIE_baichuan2_all_lora_01-03-18-59-24"
version_model="v0"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_withtype/CMeIE_baichuan2_all_lora_01-04-21-36-19"
version_model="v1"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_withtype/CMeIE_baichuan2_all_lora_01-05-23-01-49"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_withtype/CMeIE_baichuan2_all_lora_01-06-07-34-57"
version_model="v2"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_withtype/CMeIE_baichuan2_all_lora_01-07-02-59-46"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_withtype/CMeIE_baichuan2_all_lora_01-07-08-55-51"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_withtype/CMeIE_baichuan2_all_lora_01-11-14-17-18"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_withtype/CMeIE_baichuan2_all_lora_01-12-03-08-41"
version_model="v3"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_baichuan2_all_lora_01-05-13-55-53"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_baichuan2_all_lora_01-06-07-25-50"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_baichuan2_all_lora_01-05-13-55-53"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_baichuan2_all_lora_01-06-07-25-50"
version_model="v4"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_baichuan2_all_lora_01-06-17-52-49"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_baichuan2_all_lora_01-07-04-34-00"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_baichuan2_all_lora_01-06-17-52-49"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_baichuan2_all_lora_01-07-04-34-00"
version_model="v5"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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


model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/0_model_1_withtype/CMeIE_baichuan2_all_lora_01-03-18-59-24"
version_model="v0"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/1_model_1_withtype/CMeIE_baichuan2_all_lora_01-04-21-36-19"
version_model="v1"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_1_withtype/CMeIE_baichuan2_all_lora_01-05-23-01-49"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/2_model_2_withtype/CMeIE_baichuan2_all_lora_01-06-07-34-57"
version_model="v2"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_1_withtype/CMeIE_baichuan2_all_lora_01-07-02-59-46"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_2_withtype/CMeIE_baichuan2_all_lora_01-07-08-55-51"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_3_withtype/CMeIE_baichuan2_all_lora_01-11-14-17-18"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/3_model_4_withtype/CMeIE_baichuan2_all_lora_01-12-03-08-41"
version_model="v3"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_baichuan2_all_lora_01-05-13-55-53"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_baichuan2_all_lora_01-06-07-25-50"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_1_withtype/CMeIE_baichuan2_all_lora_01-05-13-55-53"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/4_model_2_withtype/CMeIE_baichuan2_all_lora_01-06-07-25-50"
version_model="v4"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_baichuan2_all_lora_01-06-17-52-49"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_baichuan2_all_lora_01-07-04-34-00"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_1_withtype/CMeIE_baichuan2_all_lora_01-06-17-52-49"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_export/5_model_2_withtype/CMeIE_baichuan2_all_lora_01-07-04-34-00"
version_model="v5"
version_type="with_type"
model_dtype="bf16"
template="baichuan2"
split_max_len=4000


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





