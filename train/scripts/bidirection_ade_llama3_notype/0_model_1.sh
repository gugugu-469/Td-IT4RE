export WANDB_DISABLED=True
now_time=$(command date +%m-%d-%H-%M-%S)
echo "now time ${now_time}"
cd ../../LLaMA-Factory
deepspeed --include localhost:2 --master_port=10722 src/train.py --deepspeed ./ds_config.json \
    --stage sft \
    --model_name_or_path ../../llm_models/Meta-Llama-3.1-8B-Instruct \
    --do_train \
    --lora_rank 16 \
    --lora_alpha 32 \
    --dataset ADE_c_to_hrt \
    --template llama3 \
    --finetuning_type lora \
    --output_dir ../../trained_models/RE_SFT_0104/0_model_1_notype/ade_llama3_all_lora_${now_time} \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 1e-4 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --preprocessing_num_workers  48 \
    --bf16 \
    --cutoff_len 8100 \
    --ddp_timeout 180000 \
    --save_total_limit 5 \
    --lora_target q_proj,v_proj


