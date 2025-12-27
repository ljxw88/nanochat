# nanochat training report

Generated: 2025-12-27 00:58:46

## Environment

### Git Information
- Branch: master
- Commit: 295be23 (dirty)
- Message: update

### Hardware
- Platform: Linux
- CPUs: 112 cores (224 logical)
- Memory: 377.0 GB
- GPUs: 1x NVIDIA H100 NVL
- GPU Memory: 93.1 GB total
- CUDA Version: 12.8
- Hourly Rate: $3.00/hour

### Software
- Python: 3.11.14
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 637,400
- Lines: 16,342
- Files: 69
- Tokens (approx): 159,350
- Dependencies (uv.lock lines): 2,218

Run started: 2025-12-27 00:58:46

---

## Tokenizer training
timestamp: 2025-12-27 01:00:08

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 77.9814
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9151
- token_bytes_std: 2.8736


## Tokenizer evaluation
timestamp: 2025-12-27 01:00:12

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 375 | 4.85 | +7.2% |
| korean | 893 | 745 | 1.20 | 721 | 1.24 | +3.2% |
| code | 1259 | 576 | 2.19 | 493 | 2.55 | +14.4% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 225 | 4.94 | +13.5% |
| fwe-train | 4208518 | 900364 | 4.67 | 856901 | 4.91 | +4.8% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1010356 | 4.86 | +4.6% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 375 | 4.85 | +3.1% |
| korean | 893 | 364 | 2.45 | 721 | 1.24 | -98.1% |
| code | 1259 | 309 | 4.07 | 493 | 2.55 | -59.5% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 225 | 4.94 | +9.6% |
| fwe-train | 4208518 | 874799 | 4.81 | 856901 | 4.91 | +2.0% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1010356 | 4.86 | +1.9% |


## Midtraining
timestamp: 2025-12-27 02:16:28

- run: d26
- device_type: 
- dtype: bfloat16
- num_iterations: -1
- max_seq_len: 2048
- device_batch_size: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- dry_run: 0
- Number of iterations: 813
- DDP world size: 1
- Minimum validation bpb: 0.3964


## Chat evaluation mid
timestamp: 2025-12-27 02:58:21

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.3603
- ARC-Challenge: 0.3123
- MMLU: 0.2973
- GSM8K: 0.0303
- HumanEval: 0.0793
- SpellingBee: 0.9844
- ChatCORE metric: 0.2312


## Chat SFT
timestamp: 2025-12-27 03:06:03

- run: d26
- source: mid
- device_type: 
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- num_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- Training rows: 22,439
- Number of iterations: 701
- Training loss: 1.0145
- Validation loss: 1.0109


## Chat evaluation sft
timestamp: 2025-12-27 03:41:51

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.3817
- ARC-Challenge: 0.3174
- MMLU: 0.3081
- GSM8K: 0.0493
- HumanEval: 0.0671
- SpellingBee: 0.9844
- ChatCORE metric: 0.2406


## Summary

- Characters: 637,400
- Lines: 16,342
- Files: 69
- Tokens (approx): 159,350
- Dependencies (uv.lock lines): 2,218

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| ARC-Challenge   | -        | 0.3123   | 0.3174   | -        |
| ARC-Easy        | -        | 0.3603   | 0.3817   | -        |
| GSM8K           | -        | 0.0303   | 0.0493   | -        |
| HumanEval       | -        | 0.0793   | 0.0671   | -        |
| MMLU            | -        | 0.2973   | 0.3081   | -        |
| ChatCORE        | -        | 0.2312   | 0.2406   | -        |

Total wall clock time: 2h43m
