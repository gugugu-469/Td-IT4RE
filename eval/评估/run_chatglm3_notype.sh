seed=2024
gpu=0
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/CMeIE-V2

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

model_1="/home/wangjiacheng/trained_models/RE_SFT/0_model_1_notype/chatglm3_all_lora_07-18-14-42-22_export_model"
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

model_1="/home/wangjiacheng/trained_models/RE_SFT/1_model_1_notype/chatglm3_all_lora_07-18-14-42-25_export_model"
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



model_1="/home/wangjiacheng/trained_models/RE_SFT/2_model_1_notype/chatglm3_all_lora_07-18-14-42-29_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/2_model_2_notype/chatglm3_all_lora_07-18-14-42-33_export_model"
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







model_1="/home/wangjiacheng/trained_models/RE_SFT/3_model_1_notype/chatglm3_all_lora_07-18-14-42-37_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/3_model_2_notype/chatglm3_all_lora_07-18-14-42-41_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/3_model_3_notype/chatglm3_all_lora_07-18-14-42-44_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/3_model_4_notype/chatglm3_all_lora_07-18-14-42-49_export_model"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/4_model_1_notype/chatglm3_all_lora_07-18-17-28-48_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/4_model_2_notype/chatglm3_all_lora_07-18-19-30-44_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/4_model_1_notype/chatglm3_all_lora_07-18-17-28-48_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/4_model_2_notype/chatglm3_all_lora_07-18-19-30-44_export_model"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/5_model_1_notype/chatglm3_all_lora_07-18-19-30-48_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/5_model_2_notype/chatglm3_all_lora_07-18-19-30-53_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/5_model_1_notype/chatglm3_all_lora_07-18-19-30-48_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/5_model_2_notype/chatglm3_all_lora_07-18-19-30-53_export_model"
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





