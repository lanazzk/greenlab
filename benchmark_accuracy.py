pip install datasets requests
cat > benchmark_accuracy.py <<'PY'
import argparse
import csv
import re
import time

import requests
from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama3:8b-instruct-fp16")
parser.add_argument("--questions", type=int, default=100)
args = parser.parse_args()

print("Loading MMLU-Pro test questions...", flush=True)
dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
questions = dataset.shuffle(seed=42).select(range(args.questions))

filename = f"results_{args.model.replace(':', '_')}.csv"
correct = 0
invalid = 0
start = time.monotonic()

with open(filename, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["question_id", "category", "expected", "predicted",
                    "correct", "raw_response", "seconds"],
    )
    writer.writeheader()

    for number, item in enumerate(questions, start=1):
        choices = "\n".join(
            f"{LETTERS[i]}. {option}"
            for i, option in enumerate(item["options"])
        )
        prompt = (
            "Solve the question. You may show your reasoning, but your final line "
            "must be exactly `Answer: X`, where X is one letter from A to J.\n\n"
            f"Question: {item['question']}\n\n{choices}\n"
        )

        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": args.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 128},
                },
                timeout=300,
            )
            response.raise_for_status()
            raw = response.json()["response"].strip()
            match = re.fullmatch(r"\s*([A-J])(?:[.)])?\s*", raw, re.I)
            predicted = match.group(1).upper() if match else ""
        except requests.RequestException as exc:
            raw = f"REQUEST ERROR: {exc}"
            predicted = ""

        is_correct = predicted == item["answer"]
        correct += is_correct
        invalid += not bool(predicted)
        elapsed = time.monotonic() - start

        writer.writerow({
            "question_id": item["question_id"],
            "category": item["category"],
            "expected": item["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "raw_response": raw,
            "seconds": round(elapsed, 2),
        })
        file.flush()
        print(
            f"{number}/{args.questions} | expected={item['answer']} "
            f"predicted={predicted or 'INVALID'} | "
            f"accuracy={correct / number:.1%} | {elapsed:.1f}s",
            flush=True,
        )

print(f"\nAccuracy: {correct}/{args.questions} = {correct / args.questions:.1%}")
print(f"Invalid responses: {invalid}")
print(f"Total time: {time.monotonic() - start:.1f}s")
print(f"Saved: {filename}")

