import sys
import os
from pathlib import Path

# Add project root to sys.path so 'src' can be imported from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()

DEFAULT_EVAL_DATA = [
    {
        "question": "What is the target latency and deployment model for Project Nebula?",
        "ground_truth": "Target latency is sub-50ms across multi-region deployments using Kubernetes on AWS and on-premise servers.",
        "expected_keywords": ["latency", "sub-50ms", "kubernetes", "aws", "nebula"]
    },
    {
        "question": "What encryption standard and key rotation schedule are used?",
        "ground_truth": "Data is encrypted at rest using AES-256 and in transit using TLS 1.3 with 90-day automated key rotation.",
        "expected_keywords": ["aes-256", "tls 1.3", "rotation", "90-day", "encryption"]
    },
    {
        "question": "What is the defined RTO and RPO for disaster recovery?",
        "ground_truth": "Disaster recovery objectives are an RTO of 15 minutes and an RPO of under 1 minute.",
        "expected_keywords": ["rto", "rpo", "15 minutes", "1 minute", "recovery"]
    }
]

def load_eval_dataset() -> List[Dict[str, Any]]:
    dataset_path = ROOT_DIR / "eval" / "eval_dataset.json"
    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            print(f"[Eval Warning] Could not read eval_dataset.json ({e}), using default evaluation set.")
    return DEFAULT_EVAL_DATA

def compute_precision(retrieved_text: str, keywords: List[str]) -> float:
    if not keywords or not retrieved_text:
        return 0.0
    matches = sum(1 for kw in keywords if kw.lower() in retrieved_text.lower())
    return matches / len(keywords)

def compute_recall(retrieved_text: str, ground_truth: str) -> float:
    if not ground_truth or not retrieved_text:
        return 0.0
    gt_words = [w.lower() for w in ground_truth.split() if len(w) > 3]
    if not gt_words:
        return 1.0
    matches = sum(1 for w in gt_words if w in retrieved_text.lower())
    return matches / len(gt_words)

def compute_groundedness(answer: str, retrieved_text: str) -> float:
    if not answer or not retrieved_text:
        return 0.0
    ans_words = [w.lower() for w in answer.split() if len(w) > 3]
    if not ans_words:
        return 1.0
    matches = sum(1 for w in ans_words if w in retrieved_text.lower())
    return min(1.0, matches / len(ans_words))

def run_evaluation():
    print("[Eval] Initializing RAG Pipeline...")
    pipeline = RAGPipeline("data")
    
    dataset = load_eval_dataset()
    print(f"\n[Eval] Running benchmark on {len(dataset)} evaluation questions...\n")
    
    results = []
    
    for i, item in enumerate(dataset, 1):
        q = item.get("question") or item.get("query", "")
        gt = item.get("ground_truth") or item.get("answer", "")
        keywords = item.get("expected_keywords") or [w for w in gt.split() if len(w) > 4]
        
        start_t = time.time()
        res = pipeline.query(q, top_k=5, top_n=2)
        latency = round(time.time() - start_t, 3)
        
        # Safely extract context strings
        ctx_list = res.get("retrieved_context", [])
        if not ctx_list and "citations" in res:
            ctx_list = [str(c) for c in res.get("citations", [])]
            
        raw_ctx = " ".join(ctx_list)
        generated_answer = res.get("answer", "")
        
        prec = compute_precision(raw_ctx, keywords)
        rec = compute_recall(raw_ctx, gt)
        grounded = compute_groundedness(generated_answer, raw_ctx)
        
        results.append({
            "question": q,
            "precision": prec,
            "recall": rec,
            "groundedness": grounded,
            "latency": latency
        })
        
        print(f"[{i}/{len(dataset)}] Q: {q[:45]}...")
        print(f"       Precision: {prec:.2f} | Recall: {rec:.2f} | Groundedness: {grounded:.2f} | Latency: {latency}s\n")

    # Aggregate Metrics
    avg_prec = sum(r["precision"] for r in results) / len(results)
    avg_rec = sum(r["recall"] for r in results) / len(results)
    avg_grounded = sum(r["groundedness"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    print("=" * 60)
    print("               FINAL EVALUATION BENCHMARK               ")
    print("=" * 60)
    print(f" Context Precision : {avg_prec * 100:.1f}%")
    print(f" Context Recall    : {avg_rec * 100:.1f}%")
    print(f" Groundedness      : {avg_grounded * 100:.1f}%")
    print(f" Average Latency   : {avg_latency:.2f} seconds")
    print("=" * 60)
    
    # Generate visualization chart
    metrics = ["Context Precision", "Context Recall", "Groundedness"]
    scores = [avg_prec * 100, avg_rec * 100, avg_grounded * 100]
    colors = ["#3b82f6", "#10b981", "#8b5cf6"]
    
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(metrics, scores, color=colors, width=0.5)
    plt.ylim(0, 115)
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Production RAG Evaluation Benchmark Results", fontsize=13, fontweight="bold")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    chart_output = ROOT_DIR / "eval_results.png"
    plt.savefig(chart_output, dpi=300)
    print(f"[Eval] Benchmark chart saved to '{chart_output.name}' successfully.")

if __name__ == "__main__":
    run_evaluation()