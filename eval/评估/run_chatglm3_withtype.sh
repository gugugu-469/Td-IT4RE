seed=2024
gpu=1
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/CMeIE-V2

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"



model_1="/home/wangjiacheng/llm_models/chatglm3-6b"
version_model="ori"
version_type="with_type"
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

model_1="/home/wangjiacheng/trained_models/RE_SFT/0_model_1_withtype/chatglm3_all_lora_07-19-04-23-21_export_model"
version_model="v0"
version_type="with_type"
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

model_1="/home/wangjiacheng/trained_models/RE_SFT/1_model_1_withtype/chatglm3_all_lora_07-19-04-23-25_export_model"
version_model="v1"
version_type="with_type"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/2_model_1_withtype/chatglm3_all_lora_07-19-04-48-18_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/2_model_2_withtype/chatglm3_all_lora_07-19-04-23-33_export_model"
version_model="v2"
version_type="with_type"
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







model_1="/home/wangjiacheng/trained_models/RE_SFT/3_model_1_withtype/chatglm3_all_lora_07-19-04-23-37_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/3_model_2_withtype/chatglm3_all_lora_07-19-04-48-38_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/3_model_3_withtype/chatglm3_all_lora_07-19-10-42-43_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/3_model_4_withtype/chatglm3_all_lora_07-19-10-42-51_export_model"
version_model="v3"
version_type="with_type"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/4_model_1_withtype/chatglm3_all_lora_07-19-10-43-00_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/4_model_2_withtype/chatglm3_all_lora_07-19-10-43-09_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/4_model_1_withtype/chatglm3_all_lora_07-19-10-43-00_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/4_model_2_withtype/chatglm3_all_lora_07-19-10-43-09_export_model"
version_model="v4"
version_type="with_type"
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




model_1="/home/wangjiacheng/trained_models/RE_SFT/5_model_1_withtype/chatglm3_all_lora_07-19-10-43-17_export_model"
model_2="/home/wangjiacheng/trained_models/RE_SFT/5_model_2_withtype/chatglm3_all_lora_07-20-03-13-53_export_model"
model_3="/home/wangjiacheng/trained_models/RE_SFT/5_model_1_withtype/chatglm3_all_lora_07-19-10-43-17_export_model"
model_4="/home/wangjiacheng/trained_models/RE_SFT/5_model_2_withtype/chatglm3_all_lora_07-20-03-13-53_export_model"
version_model="v5"
version_type="with_type"
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





