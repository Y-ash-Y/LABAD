# llm/retriever.py
"""
Given a flagged user's behavioral evidence,
retrieve the most relevant MITRE ATT&CK techniques.

The key design decision: what is the query?

Option A: embed the raw feature vector → search
  Problem: feature numbers have no semantic meaning
  "usb_connect_count: 7.3" doesn't embed well

Option B: convert features to natural language → embed → search
  WHY this works: sentence transformers understand language.
  "User connected USB 7 times, mostly after midnight" 
  semantically matches "Exfiltration Over Physical Medium"
  
  This is the right approach.
"""

import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class MITRERetriever:
    
    def __init__(self, index_dir: str = "data/mitre_index"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.index = faiss.read_index(f"{index_dir}/mitre.index")
        
        with open(f"{index_dir}/techniques.pkl", 'rb') as f:
            self.techniques = pickle.load(f)
        
        print(f"Retriever loaded: {self.index.ntotal} techniques")
    
    def features_to_natural_language(
        self, 
        user_id: str,
        user_scores: pd.Series,
        threshold: float
    ) -> str:
        """
        Convert raw behavioral features into a natural language
        description of what the user did.
        
        WHY this translation step?
        This is the bridge between your ML model and the LLM.
        The anomaly detector speaks numbers.
        The retriever and LLM speak language.
        This function is that translator.
        
        Think of it as: what would a human analyst write
        in their initial incident notes?
        """
        evidence_parts = []
        
        # ── Logon evidence ───────────────────────────────────────
        if user_scores.get('after_hours_logons', 0) > 2:
            evidence_parts.append(
                f"user logged in {user_scores['after_hours_logons']:.0f} "
                f"times outside business hours (before 7am or after 8pm)"
            )
        
        if user_scores.get('weekend_logons', 0) > 0:
            evidence_parts.append(
                f"user logged in on weekends "
                f"({user_scores['weekend_logons']:.0f} sessions)"
            )
        
        # ── USB evidence ─────────────────────────────────────────
        if user_scores.get('usb_connect_count', 0) > 3:
            after_h = user_scores.get('usb_after_hours', 0)
            evidence_parts.append(
                f"user connected USB devices "
                f"{user_scores['usb_connect_count']:.0f} times"
                + (f", including {after_h:.0f} after-hours connections"
                   if after_h > 0 else "")
            )
        
        # ── File access evidence ─────────────────────────────────
        if user_scores.get('file_access_count', 0) > 100:
            evidence_parts.append(
                f"user accessed {user_scores['file_access_count']:.0f} "
                f"files in a single day "
                f"({user_scores.get('unique_files', 0):.0f} unique files)"
            )
        
        if user_scores.get('file_after_hours', 0) > 10:
            evidence_parts.append(
                f"large volume of file access occurred after hours "
                f"({user_scores['file_after_hours']:.0f} events)"
            )
        
        if user_scores.get('exe_access_count', 0) > 2:
            evidence_parts.append(
                f"user accessed {user_scores['exe_access_count']:.0f} "
                f"executable files (unusual for this role)"
            )
        
        # ── Email evidence ───────────────────────────────────────
        if user_scores.get('external_email_count', 0) > 5:
            evidence_parts.append(
                f"user sent {user_scores['external_email_count']:.0f} "
                f"emails to external addresses"
            )
        
        if user_scores.get('emails_with_attachments', 0) > 3:
            evidence_parts.append(
                f"user sent {user_scores['emails_with_attachments']:.0f} "
                f"emails with attachments to external recipients"
            )
        
        # ── HTTP evidence ────────────────────────────────────────
        if user_scores.get('job_site_visits', 0) > 2:
            evidence_parts.append(
                f"user visited job search websites "
                f"{user_scores['job_site_visits']:.0f} times "
                f"(LinkedIn, Indeed, Glassdoor)"
            )
        
        if user_scores.get('cloud_storage_visits', 0) > 1:
            evidence_parts.append(
                f"user accessed cloud storage services "
                f"({user_scores['cloud_storage_visits']:.0f} visits to "
                f"Dropbox, Google Drive, or similar)"
            )
        
        if not evidence_parts:
            evidence_parts = [
                "user behavior deviated significantly from 30-day baseline"
            ]
        
        evidence_text = (
            f"Security alert for user {user_id}: "
            + "; ".join(evidence_parts)
            + f". Anomaly score: {user_scores.get('max_score', 0):.2f} "
            f"(threshold: {threshold:.2f})."
        )
        
        return evidence_text
    
    def retrieve(
        self, 
        query_text: str, 
        top_k: int = 5
    ) -> list:
        """
        Retrieve top-k most relevant ATT&CK techniques
        for a given behavioral evidence description.
        """
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), 
            k=top_k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            technique = self.techniques[idx].copy()
            technique['relevance_score'] = float(score)
            results.append(technique)
        
        return results
    
    def retrieve_for_user(
        self,
        user_id: str,
        user_row: pd.Series,
        threshold: float,
        top_k: int = 5
    ) -> dict:
        """
        Full retrieval pipeline for a single flagged user.
        Returns everything the LLM explainer needs.
        """
        # Step 1: Convert features to natural language
        evidence_text = self.features_to_natural_language(
            user_id, user_row, threshold
        )
        
        # Step 2: Retrieve relevant techniques
        techniques = self.retrieve(evidence_text, top_k=top_k)
        
        return {
            'user_id':      user_id,
            'evidence_text': evidence_text,
            'techniques':   techniques,
            'anomaly_score': user_row.get('max_score', 0),
            'threshold':    threshold,
        }