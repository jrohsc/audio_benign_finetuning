#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add this

DIR=`pwd`

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="moonshotai/Kimi-Audio-7B-Instruct"
PRETRAINED_MODEL_PATH=""
DATA=""

function usage() {
    echo '
Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model_path )
            shift
            PRETRAINED_MODEL_PATH=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

if [ ! -f "$DATA" ]; then
    echo "Error: DATA file does not exist"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist"
    exit 1
fi

echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "DATA: $DATA"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "start finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --model_path $PRETRAINED_MODEL_PATH \
    --data_path $DATA \
    --eval_ratio 0.05 \
    --bf16 True \
    --output_dir output/finetuned_kimi_instruct_audio_0_93 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ds_config_zero3.json \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory False