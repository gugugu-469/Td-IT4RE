seed=2024
gpu=6
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/CMeIE-V2

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"





model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/1_model_1_notype/CMeIE_glm4_all_lora_01-12-21-38-32"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="glm4"
split_max_len=4096


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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_1_notype/CMeIE_glm4_all_lora_01-13-18-48-05"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_2_notype/CMeIE_glm4_all_lora_01-14-01-22-51"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="glm4"
split_max_len=4096


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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_1_notype/CMeIE_glm4_all_lora_01-14-18-38-46"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_2_notype/CMeIE_glm4_all_lora_01-14-23-10-40"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_3_notype/CMeIE_glm4_all_lora_01-15-03-47-01"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_4_notype/CMeIE_glm4_all_lora_01-15-10-42-32"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_glm4_all_lora_01-15-23-23-03"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_glm4_all_lora_01-16-08-16-20"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_glm4_all_lora_01-15-23-23-03"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_glm4_all_lora_01-16-08-16-20"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_glm4_all_lora_01-16-23-00-35"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_glm4_all_lora_01-17-13-34-52"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_glm4_all_lora_01-16-23-00-35"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_glm4_all_lora_01-17-13-34-52"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/1_model_1_notype/CMeIE_glm4_all_lora_01-12-21-38-32"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_1_notype/CMeIE_glm4_all_lora_01-13-18-48-05"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_2_notype/CMeIE_glm4_all_lora_01-14-01-22-51"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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











model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_1_notype/CMeIE_glm4_all_lora_01-14-18-38-46"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_2_notype/CMeIE_glm4_all_lora_01-14-23-10-40"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_3_notype/CMeIE_glm4_all_lora_01-15-03-47-01"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_4_notype/CMeIE_glm4_all_lora_01-15-10-42-32"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_glm4_all_lora_01-15-23-23-03"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_glm4_all_lora_01-16-08-16-20"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_glm4_all_lora_01-15-23-23-03"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_glm4_all_lora_01-16-08-16-20"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_glm4_all_lora_01-16-23-00-35"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_glm4_all_lora_01-17-13-34-52"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_glm4_all_lora_01-16-23-00-35"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_glm4_all_lora_01-17-13-34-52"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="glm4"
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





