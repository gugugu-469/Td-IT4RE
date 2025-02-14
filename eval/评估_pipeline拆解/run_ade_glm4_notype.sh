seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

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

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_export/0_model_1_notype/CMeIE_glm4_all_lora_01-07-20-21-33"
# version_model="v0"
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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/1_model_1_notype/ade_glm4_all_lora_01-18-14-59-03"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_1_notype/ade_glm4_all_lora_01-18-20-43-02"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_2_notype/ade_glm4_all_lora_01-18-23-52-35"
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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_1_notype/ade_glm4_all_lora_01-19-03-09-21"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_2_notype/ade_glm4_all_lora_01-19-05-00-45"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_3_notype/ade_glm4_all_lora_01-19-06-51-55"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_4_notype/ade_glm4_all_lora_01-19-08-44-01"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_notype/ade_glm4_all_lora_01-19-11-15-09"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_notype/ade_glm4_all_lora_01-19-14-12-01"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_notype/ade_glm4_all_lora_01-19-11-15-09"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_notype/ade_glm4_all_lora_01-19-14-12-01"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_notype/ade_glm4_all_lora_01-19-17-46-44"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_notype/ade_glm4_all_lora_01-19-21-11-10"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_notype/ade_glm4_all_lora_01-19-17-46-44"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_notype/ade_glm4_all_lora_01-19-21-11-10"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/1_model_1_notype/ade_glm4_all_lora_01-18-14-59-03"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_1_notype/ade_glm4_all_lora_01-18-20-43-02"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_2_notype/ade_glm4_all_lora_01-18-23-52-35"
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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_1_notype/ade_glm4_all_lora_01-19-03-09-21"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_2_notype/ade_glm4_all_lora_01-19-05-00-45"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_3_notype/ade_glm4_all_lora_01-19-06-51-55"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_4_notype/ade_glm4_all_lora_01-19-08-44-01"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_notype/ade_glm4_all_lora_01-19-11-15-09"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_notype/ade_glm4_all_lora_01-19-14-12-01"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_notype/ade_glm4_all_lora_01-19-11-15-09"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_notype/ade_glm4_all_lora_01-19-14-12-01"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_notype/ade_glm4_all_lora_01-19-17-46-44"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_notype/ade_glm4_all_lora_01-19-21-11-10"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_notype/ade_glm4_all_lora_01-19-17-46-44"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_notype/ade_glm4_all_lora_01-19-21-11-10"
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



