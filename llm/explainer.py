# llm/explainer.py
"""
Takes retrieved ATT&CK techniques + behavioral evidence
and generates a structured threat report via Ollama.

WHY Ollama + local model vs OpenAI API?
Three reasons that matter for this specific project:

1. Air-gapped deployment: behavioral logs are sensitive.
   Sending them to OpenAI's servers is a data governance
   problem for any real enterprise. Local LLM = no data leaves.

2. C3iHub is a government-adjacent lab. "Fully local,
   zero external API calls" is a feature, not a limitation.

3. Reproducibility: no API rate limits, no cost per call,
   consistent behavior for evaluation.

This is the architecturally correct choice, not a compromise.
"""

import os
import requests
import json
import pandas as pd
from llm.retriever import MITRERetriever


# Configurable so the same image works locally (localhost) and inside
# docker-compose (where Ollama is reachable at http://ollama:11434).
OLLAMA_URL  = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
MODEL_NAME  = "llama3.1:8b"


def build_threat_report_prompt(retrieval_result: dict) -> str:
    """
    Structured prompt engineering for threat report generation.
    
    WHY so explicit about format?
    Unstructured LLM output is useless for a SOC workflow.
    Analysts need to scan reports quickly — structured JSON
    or markdown with clear sections is actionable.
    
    WHY "only use information from the context"?
    This is the anti-hallucination instruction.
    Without it, the LLM invents ATT&CK technique IDs,
    fabricates statistics, or adds irrelevant techniques.
    Grounding it in retrieved context is the core RAG principle.
    """
    # Format retrieved techniques for context
    technique_context = ""
    for i, tech in enumerate(retrieval_result['techniques'], 1):
        technique_context += (
            f"\n[{i}] {tech['id']} — {tech['name']}\n"
            f"    Tactics: {', '.join(tech['tactics'])}\n"
            f"    Description: {tech['description'][:300]}\n"
            f"    Detection: {tech['detection'][:200]}\n"
        )
    
    prompt = f"""You are a cybersecurity analyst at a Security Operations Center (SOC).
A behavioral anomaly detection system has flagged a user for potential insider threat activity.
Your job is to write a concise, actionable threat report.

BEHAVIORAL EVIDENCE:
{retrieval_result['evidence_text']}

ANOMALY SCORE: {retrieval_result['anomaly_score']:.2f} (threshold: {retrieval_result['threshold']:.2f})

RELEVANT MITRE ATT&CK TECHNIQUES FROM KNOWLEDGE BASE:
{technique_context}

INSTRUCTIONS:
- Only reference techniques and information present in the context above
- Do not invent statistics or technique IDs not shown above
- Be specific and concise — a SOC analyst will read this in 30 seconds
- Write in plain English, not jargon

Write a threat report with exactly these sections:

## THREAT SUMMARY
One sentence describing the most likely threat scenario.

## SEVERITY
One of: LOW / MEDIUM / HIGH
Followed by one sentence justification.

## BEHAVIORAL EVIDENCE
Bullet points of the specific suspicious behaviors observed.

## ATT&CK MAPPING
List the most relevant technique(s) from the context, with IDs.

## RECOMMENDED ACTIONS
2-3 specific, immediate actions for the SOC analyst.

## CONFIDENCE
One of: LOW / MEDIUM / HIGH
Brief explanation of what would increase or decrease confidence."""
    
    return prompt


def generate_report(prompt: str, max_tokens: int = 600) -> str:
    """
    Call Ollama API to generate threat report.
    
    WHY stream=False?
    We want the complete report before doing anything with it.
    Streaming is useful for UI, not for batch evaluation.
    """
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,   
            # WHY low temperature (0.1)?
            # We want deterministic, factual reports.
            # High temperature = creative but unreliable.
            # Threat reports need consistency, not creativity.
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120  # 2 min timeout — 8B model can be slow
        )
        response.raise_for_status()
        return response.json()['response']
    
    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Ollama is not running. "
            "Start it with: ollama serve"
        )
    except Exception as e:
        return f"ERROR: {str(e)}"


class ThreatReportGenerator:
    
    def __init__(self, index_dir: str = "data/mitre_index"):
        self.retriever = MITRERetriever(index_dir)
    
    def generate_for_user(
        self,
        user_id: str,
        user_row: pd.Series,
        threshold: float
    ) -> dict:
        """
        Full pipeline: user features → retrieval → LLM → report
        """
        # Step 1: Retrieve relevant techniques
        retrieval = self.retriever.retrieve_for_user(
            user_id, user_row, threshold
        )
        
        # Step 2: Build prompt
        prompt = build_threat_report_prompt(retrieval)
        
        # Step 3: Generate report
        print(f"  Generating report for {user_id}...")
        report_text = generate_report(prompt)
        
        return {
            'user_id':       user_id,
            'anomaly_score': retrieval['anomaly_score'],
            'evidence_text': retrieval['evidence_text'],
            'techniques':    retrieval['techniques'],
            'prompt':        prompt,
            'report':        report_text,
        }
    
    def generate_batch(
        self,
        user_scores_df: pd.DataFrame,
        threshold: float,
        top_n: int = 10
    ) -> list:
        """
        Generate reports for top-N most anomalous users.
        
        WHY top_N only?
        In a real SOC, you triage the highest-risk users first.
        Generating reports for all 256 users would take ~30 minutes
        and flood analysts with noise.
        Top 10 is realistic for daily SOC triage.
        """
        # Get top N by anomaly score
        top_users = user_scores_df.nlargest(top_n, 'max_score')
        
        reports = []
        for _, row in top_users.iterrows():
            result = self.generate_for_user(
                user_id   = row['user'],
                user_row  = row,
                threshold = threshold
            )
            result['is_malicious'] = row.get('is_malicious', -1)
            reports.append(result)
            
            # Print report immediately so you can see progress
            print(f"\n{'='*60}")
            print(f"USER: {result['user_id']} | "
                  f"Score: {result['anomaly_score']:.2f} | "
                  f"{'MALICIOUS' if result['is_malicious'] else 'BENIGN'}")
            print(f"{'='*60}")
            print(result['report'])
        
        return reports
    