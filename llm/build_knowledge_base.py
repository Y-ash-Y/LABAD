# llm/build_knowledge_base.py
"""
Builds the FAISS vector index over MITRE ATT&CK techniques.

WHY FAISS over a simple keyword search?
Keyword search misses semantic similarity.
Example: "user copied files to USB drive" won't match
"Exfiltration Over Physical Medium" via keyword.
But sentence embeddings capture the semantic meaning —
"copying to USB" and "physical medium exfiltration" 
are close in embedding space.

WHY sentence-transformers/all-MiniLM-L6-v2?
- 384-dimensional embeddings — small, fast, good quality
- Runs entirely on CPU — no GPU needed
- Trained on semantic similarity — exactly our use case
- 80MB model — downloads once, cached locally
"""

import json
import requests
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────
MITRE_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
INDEX_DIR = "data/mitre_index"
os.makedirs(INDEX_DIR, exist_ok=True)


def download_mitre_attack() -> list:
    """
    Download MITRE ATT&CK STIX data from GitHub.
    
    WHY STIX format? It's the standard structured threat
    intelligence format. MITRE publishes their entire
    knowledge base as open STIX JSON.
    
    We extract only 'attack-pattern' objects — these are
    the individual techniques (T1052, T1078, etc.)
    """
    mitre_path = f"{INDEX_DIR}/enterprise-attack.json"
    
    if os.path.exists(mitre_path):
        print("MITRE ATT&CK already downloaded, loading...")
        with open(mitre_path, 'r') as f:
            data = json.load(f)
    else:
        print("Downloading MITRE ATT&CK (~50MB)...")
        response = requests.get(MITRE_URL, stream=True)
        total = int(response.headers.get('content-length', 0))
        
        content = b""
        with tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                pbar.update(len(chunk))
        
        data = json.loads(content)
        with open(mitre_path, 'w') as f:
            json.dump(data, f)
        print("Downloaded and saved.")
    
    # Extract attack patterns (techniques)
    techniques = []
    for obj in data['objects']:
        if obj.get('type') != 'attack-pattern':
            continue
        if obj.get('revoked', False):
            continue  # skip deprecated techniques
        
        # Extract ATT&CK ID (e.g. T1052)
        attack_id = None
        for ref in obj.get('external_references', []):
            if ref.get('source_name') == 'mitre-attack':
                attack_id = ref.get('external_id')
                break
        
        if not attack_id:
            continue
        
        # Extract tactic (kill chain phase)
        tactics = [
            phase['phase_name'].replace('-', ' ').title()
            for phase in obj.get('kill_chain_phases', [])
            if phase.get('kill_chain_name') == 'mitre-attack'
        ]
        
        # Build rich text for embedding
        # WHY include name + description + detection?
        # Richer context = better semantic matching
        name        = obj.get('name', '')
        description = obj.get('description', '')[:500]  # truncate long descriptions
        detection   = obj.get('x_mitre_detection', '')[:300]
        
        full_text = (
            f"Technique: {name}. "
            f"Tactics: {', '.join(tactics)}. "
            f"Description: {description} "
            f"Detection: {detection}"
        )
        
        techniques.append({
            'id':          attack_id,
            'name':        name,
            'tactics':     tactics,
            'description': description,
            'detection':   detection,
            'full_text':   full_text,
        })
    
    print(f"Extracted {len(techniques)} MITRE ATT&CK techniques")
    return techniques


def add_cert_scenarios(techniques: list) -> list:
    """
    Add CERT-specific scenario descriptions as additional
    knowledge base entries.
    
    WHY add these manually?
    MITRE ATT&CK describes techniques in general terms.
    Our synthetic CERT scenarios have specific behavioral
    signatures. Adding explicit mappings helps the retriever
    surface the right techniques for our exact feature set.
    
    This is domain knowledge engineering — a real contribution
    that generic RAG systems don't do.
    """
    cert_entries = [
        {
            'id':      'CERT-USB-EXFIL',
            'name':    'USB Exfiltration Pattern',
            'tactics': ['Exfiltration'],
            'description': (
                'Insider threat actor connects USB storage devices '
                'multiple times per day, often during after-hours periods. '
                'High volume of file access events precede USB connections. '
                'Characteristic signature: USB connect count spikes 3-8x '
                'above baseline, combined with after-hours login activity. '
                'Maps to MITRE T1052 Exfiltration Over Physical Medium.'
            ),
            'detection': (
                'Monitor USB connection frequency per user per day. '
                'Flag users with USB connections outside business hours. '
                'Correlate USB events with file access volume spikes.'
            ),
            'full_text': '',  # filled below
        },
        {
            'id':      'CERT-EMAIL-EXFIL',
            'name':    'Email Exfiltration Pattern',
            'tactics': ['Collection', 'Exfiltration'],
            'description': (
                'Insider sends abnormal volume of emails to external '
                'addresses, often with attachments containing sensitive data. '
                'External email ratio rises sharply above user baseline. '
                'Attachments and email size increase significantly. '
                'Often correlated with job site browsing indicating '
                'intent to leave organization. '
                'Maps to MITRE T1114 Email Collection and T1048 '
                'Exfiltration Over Alternative Protocol.'
            ),
            'detection': (
                'Track ratio of external to internal emails per user. '
                'Alert on sudden increases in emails with attachments. '
                'Correlate with job site visits on same days.'
            ),
            'full_text': '',
        },
        {
            'id':      'CERT-FILE-THEFT',
            'name':    'Bulk File Access Theft Pattern',
            'tactics': ['Collection', 'Exfiltration'],
            'description': (
                'Insider accesses abnormally large number of files '
                'in short time period, often targeting documents and '
                'spreadsheets outside their normal working set. '
                'File access count spikes 10-50x above daily baseline. '
                'Activity often occurs during after-hours periods. '
                'May be followed by USB or email exfiltration. '
                'Maps to MITRE T1119 Automated Collection and '
                'T1005 Data from Local System.'
            ),
            'detection': (
                'Establish per-user daily file access baseline. '
                'Flag sessions with file access 5x above 30-day average. '
                'Monitor for access to files outside normal directory paths.'
            ),
            'full_text': '',
        },
        {
            'id':      'CERT-AFTER-HOURS',
            'name':    'After-Hours Access Pattern',
            'tactics': ['Defense Evasion', 'Collection'],
            'description': (
                'Insider deliberately performs sensitive operations '
                'outside normal business hours to avoid detection. '
                'Login activity between 10pm and 6am is a strong signal. '
                'Combined with other exfiltration indicators creates '
                'high-confidence insider threat signature. '
                'Maps to MITRE T1078 Valid Accounts used outside '
                'normal operating patterns.'
            ),
            'detection': (
                'Track login timestamps relative to user baseline hours. '
                'Flag after-hours access when combined with file, USB, '
                'or email anomalies. Consider timezone and shift patterns.'
            ),
            'full_text': '',
        },
        {
            'id':      'CERT-RECON',
            'name':    'Pre-Departure Reconnaissance Pattern',
            'tactics': ['Discovery', 'Collection'],
            'description': (
                'Insider approaching departure visits job search websites '
                'and cloud storage services while simultaneously increasing '
                'data collection activities. Job site visits on LinkedIn, '
                'Indeed, Glassdoor combined with data exfiltration attempts. '
                'Cloud storage access to Dropbox, Google Drive, OneDrive '
                'for data staging. Maps to MITRE T1119 Automated Collection '
                'with pre-departure behavioral indicators.'
            ),
            'detection': (
                'Monitor web browsing for job site visits. '
                'Flag cloud storage access outside normal usage. '
                'Correlate browsing patterns with data access anomalies.'
            ),
            'full_text': '',
        },
    ]
    
    # Fill full_text for embedding
    for entry in cert_entries:
        entry['full_text'] = (
            f"Technique: {entry['name']}. "
            f"Tactics: {', '.join(entry['tactics'])}. "
            f"Description: {entry['description']} "
            f"Detection: {entry['detection']}"
        )
    
    combined = techniques + cert_entries
    print(f"Added {len(cert_entries)} CERT-specific entries")
    print(f"Total knowledge base: {len(combined)} entries")
    return combined


def build_faiss_index(techniques: list) -> tuple:
    """
    Embed all techniques and build FAISS index.
    
    WHY IVF (Inverted File) index?
    For ~800 techniques, a flat index (exact search) is fast enough.
    We use IndexFlatIP (inner product) with normalized vectors,
    which is equivalent to cosine similarity search.
    
    WHY cosine similarity?
    We want semantic similarity regardless of text length.
    Cosine similarity is length-normalized dot product —
    a short technique description and a long one can still
    score high similarity if they're semantically close.
    """
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # First run downloads ~80MB model, subsequent runs use cache
    
    print("Embedding techniques...")
    texts = [t['full_text'] for t in techniques]
    
    # Batch embedding — faster than one at a time
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    
    # embeddings shape: (n_techniques, 384)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Build FAISS flat index with inner product
    # (cosine similarity on normalized vectors = inner product)
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    
    print(f"FAISS index built: {index.ntotal} vectors")
    return index, embeddings


def save_index(
    index, 
    techniques: list, 
    embeddings: np.ndarray
):
    """Save everything needed for retrieval at inference time."""
    faiss.write_index(index, f"{INDEX_DIR}/mitre.index")
    
    with open(f"{INDEX_DIR}/techniques.pkl", 'wb') as f:
        pickle.dump(techniques, f)
    
    np.save(f"{INDEX_DIR}/embeddings.npy", embeddings)
    
    print(f"\nSaved to {INDEX_DIR}/:")
    print(f"  mitre.index      — FAISS index")
    print(f"  techniques.pkl   — technique metadata")
    print(f"  embeddings.npy   — raw embeddings")


def main():
    print("="*50)
    print("Building MITRE ATT&CK Knowledge Base")
    print("="*50)
    
    techniques = download_mitre_attack()
    techniques = add_cert_scenarios(techniques)
    index, embeddings = build_faiss_index(techniques)
    save_index(index, techniques, embeddings)
    
    # Quick sanity check
    print("\nSanity check — searching for 'USB drive file copy':")
    model   = SentenceTransformer('all-MiniLM-L6-v2')
    query   = model.encode(
        ["user copied large number of files to USB drive after hours"],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    scores, indices = index.search(query.astype(np.float32), k=3)
    
    for score, idx in zip(scores[0], indices[0]):
        t = techniques[idx]
        print(f"  [{t['id']}] {t['name']} "
              f"(similarity: {score:.3f})")
    
    print("\nKnowledge base ready.")


if __name__ == '__main__':
    main()