seed=2024
gpu=0
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"




model_1="/home/wangjiacheng/llm_models/chatglm3-6b"
version_model="ori"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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

model_1="/home/wangjiacheng/trained_models/RE_SFT/0_model_1_notype/chatglm3_ade_all_lora_07-22-10-50-26_export_model"
version_model="v0"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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

model_1="/home/wangjiacheng/trained_models/RE_SFT/1_model_1_notype/chatglm3_ade_all_lora_07-22-10-50-31_export_model"
version_model="v1"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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



model_1="/home/wangjiacheng/trained_models/RE_SFT/2_model_1_notype/chatglm3_ade_all_lora_07-22-10-50-35_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/2_model_2_notype/chatglm3_ade_all_lora_07-22-16-23-15_export_model"
version_model="v2"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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







model_1="/home/wangjiacheng/trained_models/RE_SFT/3_model_1_notype/chatglm3_ade_all_lora_07-22-16-23-19_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/3_model_2_notype/chatglm3_ade_all_lora_07-22-16-23-23_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/3_model_3_notype/chatglm3_ade_all_lora_07-22-16-23-28_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/3_model_4_notype/chatglm3_ade_all_lora_07-22-16-23-32_export_model"
version_model="v3"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/4_model_1_notype/chatglm3_ade_all_lora_07-22-16-23-36_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/4_model_2_notype/chatglm3_ade_all_lora_07-22-17-47-28_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/4_model_1_notype/chatglm3_ade_all_lora_07-22-16-23-36_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/4_model_2_notype/chatglm3_ade_all_lora_07-22-17-47-28_export_model"
version_model="v4"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/5_model_1_notype/chatglm3_ade_all_lora_07-22-17-47-32_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/5_model_2_notype/chatglm3_ade_all_lora_07-22-17-47-35_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/5_model_1_notype/chatglm3_ade_all_lora_07-22-17-47-32_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/5_model_2_notype/chatglm3_ade_all_lora_07-22-17-47-35_export_model"
version_model="v5"
version_type="no_type"
model_dtype="fp16"
template="chatglm3"
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





