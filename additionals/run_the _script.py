"""
build_pandas_datasets.py: 
# first time (full run)
python build_pandas_datasets.py --repo-url https://github.com/pandas-dev/pandas

# reuse an existing clone
python build_pandas_datasets.py --no-clone

# useful options
#   --max-commits 4000       # more/less repair mining
#   --depth 0                # 0=full history; >0 shallow clone
#   --steps summarization signature search repair   # subset
#   --skip-clean             # only raw JSONLs (no clean_data)


# install deps
pip install "transformers>=4.44" datasets sentencepiece accelerate evaluate sacrebleu rouge-score

# train (default model: Salesforce/codet5p-220m)
python train_multitask_codet5.py \
  --model_name Salesforce/codet5p-220m \
  --train_ratio 0.9 --seed 42 \
  --batch_size 8 --grad_accum 2 \
  --lr 2e-5 --epochs 3 \
  --max_input_len 768 --max_target_len 256 \
  --save_dir ./outputs/codet5p_mt_pandas

# You’ll get: checkpoints, logs, and task-wise eval (EM for signature, ROUGE/BLEU for 
  summarization & repair, BLEU for code search text match).

  
# in your venv
pip install GitPython tqdm rapidfuzz
pip install "transformers>=4.44" datasets sentencepiece accelerate evaluate sacrebleu rouge-score


# B) (optional) sanity check
python validate_datasets.py

How to run (choose your mode)

LoRA (recommended on RTX 3080, best quality vs. compute):

pip install peft bitsandbytes  # bitsandbytes optional for LoRA; required for QLoRA
python train_multitask_codet5.py \
  --model_name Salesforce/codet5p-220m \
  --peft lora --gradient_checkpointing \
  --limit_per_task 2000 --train_ratio 0.9 \
  --batch_size 8 --grad_accum 2 --lr 2e-5 --epochs 3 \
  --max_input_len 768 --max_target_len 256 \
  --save_dir outputs/codet5p220m_lora_bal2k


QLoRA (lowest VRAM; good if you hit OOM):

python train_multitask_codet5.py \
  --model_name Salesforce/codet5p-220m \
  --peft qlora --gradient_checkpointing \
  --limit_per_task 2000 --train_ratio 0.9 \
  --batch_size 8 --grad_accum 4 --lr 1.5e-4 --epochs 3 \
  --save_dir outputs/codet5p220m_qlora_bal2k


(QLoRA uses 4-bit quantization + LoRA; a slightly higher LR like 1e-4 ~ 2e-4 often works better.)

Full fine-tune (highest quality but heaviest):

python train_multitask_codet5.py \
  --model_name Salesforce/codet5p-220m \
  --peft none \
  --batch_size 4 --grad_accum 4 --lr 1e-5 --epochs 3 \
  --save_dir outputs/codet5p220m_full


Bigger base (if VRAM allows, try for your ablation):

python train_multitask_codet5.py \
  --model_name Salesforce/codet5p-770m \
  --peft lora --gradient_checkpointing \
  --batch_size 4 --grad_accum 4 --lr 1.5e-5 --epochs 3 \
  --save_dir outputs/codet5p770m_lora





"""