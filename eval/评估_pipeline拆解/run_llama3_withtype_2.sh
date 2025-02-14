seed=12341
gpu=2
dataset_dir=/home/wangjiacheng/面向RE任务微调LLM/datas/pipeline拆解/ADE

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"







model_1="/home/wangjiacheng/trained_models/RE_SFT_0104_pipeline_export/2_model_1_withtype/ade_llama3_all_lora_01-20-23-29-36"
model_2="/home/wangjiacheng/trained_models/RE_SFT_0104_pipeline_export/2_model_2_withtype/ade_llama3_all_lora_02-01-20-35-20"
version_model="v2"
version_type="with_type"
model_dtype="fp16"
template="llama3"
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








model_1="/home/wangjiacheng/trained_models/RE_SFT_0104_pipeline_export/2_model_1_withtype/ade_llama3_all_lora_01-20-23-29-36"
model_2="/home/wangjiacheng/trained_models/RE_SFT_0104_pipeline_export/2_model_2_withtype/ade_llama3_all_lora_02-01-20-35-20"
version_model="v2"
version_type="with_type"
model_dtype="fp16"
template="llama3"
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





