seed=2024
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



model_1="/home/wangjiacheng/llm_models/glm-4-9b-chat"
version_model="ori"
version_type="with_type"
model_dtype="bf16"
template="glm4"
split_max_len=2000


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

# model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_export/0_model_1_withtype/CMeIE_glm4_all_lora_01-03-14-00-39"
# version_model="v0"
# version_type="with_type"
# model_dtype="bf16"
# template="glm4"
# split_max_len=2000


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

model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/1_model_1_withtype/ade_glm4_all_lora_01-26-23-23-29"
version_model="v1"
version_type="with_type"
model_dtype="bf16"
template="glm4"
split_max_len=2000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_1_withtype/ade_glm4_all_lora_01-27-02-56-04"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/2_model_2_withtype/ade_glm4_all_lora_01-27-04-50-04"
version_model="v2"
version_type="with_type"
model_dtype="bf16"
template="glm4"
split_max_len=2000


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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_1_withtype/ade_glm4_all_lora_01-27-06-58-29"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_2_withtype/ade_glm4_all_lora_01-27-08-11-51"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_3_withtype/ade_glm4_all_lora_01-27-09-25-49"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/3_model_4_withtype/ade_glm4_all_lora_01-27-10-38-44"
version_model="v3"
version_type="with_type"
model_dtype="bf16"
template="glm4"
split_max_len=2000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_withtype/ade_glm4_all_lora_01-27-12-13-52"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_withtype/ade_glm4_all_lora_01-27-04-23-08"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_1_withtype/ade_glm4_all_lora_01-27-12-13-52"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/4_model_2_withtype/ade_glm4_all_lora_01-27-04-23-08"
version_model="v4"
version_type="with_type"
model_dtype="bf16"
template="glm4"
split_max_len=2000


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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_withtype/ade_glm4_all_lora_01-27-05-52-08"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_withtype/ade_glm4_all_lora_01-27-07-20-32"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_1_withtype/ade_glm4_all_lora_01-27-05-52-08"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models//RE_SFT_0104_pipeline_export/5_model_2_withtype/ade_glm4_all_lora_01-27-07-20-32"
version_model="v5"
version_type="with_type"
model_dtype="bf16"
template="glm4"
split_max_len=2000


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





