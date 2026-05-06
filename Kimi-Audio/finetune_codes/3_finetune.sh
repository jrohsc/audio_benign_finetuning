# module load cuda/12.6

bash finetune_ds.sh \
    --model_path "output/pretrained_hf_instruct" \
    --data "benign_close_to_harmful_0_93_semantic_codes.jsonl"