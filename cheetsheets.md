cd greenlabproject/
source .venv/bin/activate

# EXPERIMENT RUNNER
python experiment-runner/experiment-runner/RunnerConfig.py

# INSPECT VIEWER
inspect view --log-dir logs

# LLAMA 3.2 3B FP16 — MMLU-PRO — 100 QUESTIONS
energibridge \
  --output results/llama3.2-3b-fp16/mmlu_pro/energy.csv \
  -- \
inspect eval inspect_evals/mmlu_pro \
  --model ollama/llama3.2:3b-instruct-fp16 \
  --limit 1 \
  --sample-shuffle 42 \
  -T shuffle=false \
  --max-connections 1 \
  --temperature 0 \
  --max-tokens 2048 \
  --max-retries 0 \
  --log-dir logs


# LLAMA 3.2 3B FP16 — GSM8K — 100 QUESTIONS
energibridge \
  --output results/llama3.2-3b-fp16/gsm8k/energy.csv \
  -- \
  inspect eval inspect_evals/gsm8k \
    --model ollama/llama3.2:3b-instruct-fp16 \
    --limit 100 \
    --sample-shuffle 42 \
    --max-connections 1 \
    --temperature 0 \
    --max-retries 0 \
    --log-dir logs


# LLAMA 3.2 3B FP16 — RACE-H — 100 QUESTIONS
energibridge \
  --output results/llama3.2-3b-fp16/race_h/energy.csv \
  -- \
  inspect eval inspect_evals/race_h \
    --model ollama/llama3.2:3b-instruct-fp16 \
    --limit 100 \
    --sample-shuffle 42 \
    --max-connections 1 \
    --temperature 0 \
    --max-retries 0 \
    --log-dir logs