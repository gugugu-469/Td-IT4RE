seed=2024
gpu=7
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/CMeIE-V2

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/1_model_1_notype/CMeIE_intern2_all_lora_01-12-21-42-32"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_1_notype/CMeIE_intern2_all_lora_01-13-20-40-06"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_2_notype/CMeIE_intern2_all_lora_01-14-03-47-17"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_1_notype/CMeIE_intern2_all_lora_01-14-22-39-32"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_2_notype/CMeIE_intern2_all_lora_01-15-03-34-54"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_3_notype/CMeIE_intern2_all_lora_01-15-08-36-11"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_4_notype/CMeIE_intern2_all_lora_01-15-15-59-23"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_intern2_all_lora_01-16-05-59-39"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_intern2_all_lora_01-16-15-22-51"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_intern2_all_lora_01-16-05-59-39"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_intern2_all_lora_01-16-15-22-51"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_intern2_all_lora_01-17-07-28-04"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_intern2_all_lora_01-17-23-24-29"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_intern2_all_lora_01-17-07-28-04"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_intern2_all_lora_01-17-23-24-29"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/1_model_1_notype/CMeIE_intern2_all_lora_01-12-21-42-32"
version_model="v1"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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



model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_1_notype/CMeIE_intern2_all_lora_01-13-20-40-06"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/2_model_2_notype/CMeIE_intern2_all_lora_01-14-03-47-17"
version_model="v2"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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







model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_1_notype/CMeIE_intern2_all_lora_01-14-22-39-32"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_2_notype/CMeIE_intern2_all_lora_01-15-03-34-54"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_3_notype/CMeIE_intern2_all_lora_01-15-08-36-11"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/3_model_4_notype/CMeIE_intern2_all_lora_01-15-15-59-23"
version_model="v3"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_intern2_all_lora_01-16-05-59-39"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_intern2_all_lora_01-16-15-22-51"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_1_notype/CMeIE_intern2_all_lora_01-16-05-59-39"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/4_model_2_notype/CMeIE_intern2_all_lora_01-16-15-22-51"
version_model="v4"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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




model_1="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_intern2_all_lora_01-17-07-28-04"
model_2="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_intern2_all_lora_01-17-23-24-29"
model_3="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_1_notype/CMeIE_intern2_all_lora_01-17-07-28-04"
model_4="/home/wangjiacheng/mnt_mine_device_1/trained_models/RE_SFT_0104_pipeline_export/5_model_2_notype/CMeIE_intern2_all_lora_01-17-23-24-29"
version_model="v5"
version_type="no_type"
model_dtype="bf16"
template="intern2"
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





