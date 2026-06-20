# eval/run_week3.py
"""
Week 3 end-to-end pipeline.
Requires: Ollama running with llama3.1:8b pulled
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import json
import os

def check_ollama():
    """Verify Ollama is running before starting."""
    import requests
    # Derive the tags endpoint from the same OLLAMA_URL the explainer uses,
    # so this check follows the host into docker-compose (service "ollama").
    ollama_url = os.environ.get(
        "OLLAMA_URL", "http://localhost:11434/api/generate"
    )
    tags_url = ollama_url.replace("/api/generate", "/api/tags")
    try:
        r = requests.get(tags_url, timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        if not any('llama3.1' in m for m in models):
            print("ERROR: llama3.1:8b not found.")
            print("Run: ollama pull llama3.1:8b")
            return False
        print(f"Ollama running. Available models: {models}")
        return True
    except:
        print("ERROR: Ollama not running.")
        print("Start with: ollama serve")
        return False


def main():
    print("="*60)
    print("LABAD — Week 3: RAG Pipeline + LLM Threat Reports")
    print("="*60)
    
    # ── 0. Check Ollama ──────────────────────────────────────────
    if not check_ollama():
        return
    
    # ── 1. Build knowledge base (skip if exists) ─────────────────
    if not os.path.exists("data/mitre_index/mitre.index"):
        print("\nBuilding MITRE ATT&CK knowledge base...")
        os.system("python llm/build_knowledge_base.py")
    else:
        print("\nKnowledge base already exists, skipping build.")
    
    # ── 2. Load Week 2 results ───────────────────────────────────
    print("\nLoading Week 2 anomaly scores...")
    user_scores = pd.read_csv("data/processed/user_scores.csv")
    threshold   = float(np.load("data/processed/threshold.npy")[0])
    
    print(f"Loaded {len(user_scores)} user scores")
    print(f"NP threshold: {threshold:.4f}")
    print(f"Users above threshold: "
          f"{(user_scores['max_score'] > threshold).sum()}")
    
    # ── 3. Generate threat reports ───────────────────────────────
    from llm.explainer import ThreatReportGenerator
    
    generator = ThreatReportGenerator()
    
    print("\nGenerating threat reports for top 10 users...")
    print("(Each report takes ~15-30 seconds on M4)\n")
    
    reports = generator.generate_batch(
        user_scores_df = user_scores,
        threshold      = threshold,
        top_n          = 10
    )
    
    # ── 4. Save reports ──────────────────────────────────────────
    os.makedirs("data/reports", exist_ok=True)
    
    # Save full reports as JSON
    reports_json = []
    for r in reports:
        reports_json.append({
            'user_id':       r['user_id'],
            'anomaly_score': r['anomaly_score'],
            'is_malicious':  int(r['is_malicious']),
            'evidence':      r['evidence_text'],
            'techniques':    [
                {'id': t['id'], 'name': t['name']} 
                for t in r['techniques']
            ],
            'report':        r['report'],
        })
    
    with open("data/reports/threat_reports.json", 'w') as f:
        json.dump(reports_json, f, indent=2)
    
    # Save human-readable markdown
    with open("data/reports/threat_reports.md", 'w') as f:
        f.write("# LABAD Threat Reports\n\n")
        for r in reports:
            status = "🔴 MALICIOUS" if r['is_malicious'] else "🔵 BENIGN"
            f.write(f"## {r['user_id']} — {status}\n")
            f.write(f"**Anomaly Score:** {r['anomaly_score']:.2f}\n\n")
            f.write(f"**Evidence:** {r['evidence_text']}\n\n")
            f.write(r['report'])
            f.write("\n\n---\n\n")
    
    print(f"\nReports saved to data/reports/")
    
    # ── 5. Quick evaluation ──────────────────────────────────────
    print("\n" + "="*40)
    print("Top 10 Alert Summary")
    print("="*40)
    
    for r in reports:
        status = "MALICIOUS" if r['is_malicious'] else "BENIGN  "
        print(f"  {r['user_id']} | "
              f"Score: {r['anomaly_score']:6.2f} | "
              f"{status} | "
              f"Techniques: "
              f"{', '.join(t['id'] for t in r['techniques'][:2])}")
    
    malicious_in_top10 = sum(1 for r in reports if r['is_malicious'])
    print(f"\nMalicious users in top 10: {malicious_in_top10}/10")
    print(f"Precision@10: {malicious_in_top10/10*100:.0f}%")
    print("\nWeek 3 complete.")

if __name__ == '__main__':
    main()