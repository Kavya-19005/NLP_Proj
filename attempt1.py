# This script performs record linkage (duplicate detection) on the FEBRL-1 dataset using a combination of
# traditional string similarity metrics and semantic similarity from SBERT embeddings. It blocks records 
# based on given name prefixes to reduce comparisons, computes similarity features, and uses a supervised 
# Logistic Regression model to classify record pairs as matches or non-matches. Finally, it evaluates 
# model performance using precision, recall, and F1-score.

import recordlinkage
import pandas as pd
import numpy as np
from recordlinkage.datasets import load_febrl1
from recordlinkage import precision, recall
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- A. Data Loading and Preparation (CRM Proxy) ---

# Load FEBRL 1 data and true matches
df = load_febrl1()
true_matches = load_febrl1(return_links=True)[1] 

print(f"Total Records: {len(df)}")
print(f"True Duplicate Pairs: {len(true_matches)}")
print("-" * 30)

# --- 1. Prepare Text for SBERT (Simulating Name, Email, Company) ---

def create_contact_text(record):
    """Combines FEBRL fields into a single text string for SBERT encoding."""
    name = f"{record['given_name']} {record['surname']}".strip()
    id_field = str(record.get('soc_sec_id', '')).replace('.0', '')
    address = record.get('address_1', '')
    
    return f"Name: {name} | ID: {id_field} | Address: {address}"

df['contact_text'] = df.apply(create_contact_text, axis=1)

# --- 2. Encode all records using Sentence-Transformers ---
# Note: Model download/caching might take a moment on the first run.
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = df['contact_text'].tolist()
print("Encoding records with SBERT...")
embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False) 
print("Encoding complete.")
print("-" * 30)

# -------------------------------------------------------------------
# B. Blocking (Candidate Selection)
# -------------------------------------------------------------------

indexer = recordlinkage.Index()

# Create the prefix column explicitly and block on it
df['given_name_prefix'] = df['given_name'].str[:2]
indexer.block(left_on='given_name_prefix')

candidate_links = indexer.index(df)

print(f"Candidate Pairs after Blocking: {len(candidate_links):,}")
print("-" * 30)

# -------------------------------------------------------------------
# C. Comparison (Feature Generation: String Metrics + SBERT)
# -------------------------------------------------------------------

compare_cl = recordlinkage.Compare()

# 1. Traditional String Metrics
compare_cl.string('given_name', 'given_name', method='jarowinkler', label='name_jw')
compare_cl.string('surname', 'surname', method='jarowinkler', label='surname_jw')
compare_cl.exact('date_of_birth', 'date_of_birth', label='dob_exact')
compare_cl.string('address_1', 'address_1', method='damerau_levenshtein', threshold=0.8, label='address_dl')

features = compare_cl.compute(candidate_links, df)

# --- Add SBERT Similarity as a new Feature ---
sbert_scores = []
df_index_map = {idx: i for i, idx in enumerate(df.index)}

for id1, id2 in candidate_links:
    emb1 = embeddings[df_index_map[id1]]
    emb2 = embeddings[df_index_map[id2]]
    
    sim = util.cos_sim(emb1, emb2).item()
    sbert_scores.append(sim)

sbert_series = pd.Series(sbert_scores, index=candidate_links, name='sbert_sim')
features = pd.concat([features, sbert_series], axis=1)

print(f"Features created for {len(features)} candidate pairs.")
print(f"Features: {features.columns.tolist()}")
print("-" * 30)

# -------------------------------------------------------------------
# D. Classification (Supervised Learning)
# -------------------------------------------------------------------

# 1. Create Training Labels (y) - THE ROBUST FIX
# Check if each candidate pair exists in the true_matches MultiIndex
y_true = np.in1d(candidate_links, true_matches)

# 2. Split data for robust evaluation
X_train, X_test, y_train, y_test = train_test_split(
    features, y_true, test_size=0.3, random_state=42, stratify=y_true
)

# 3. Train the Classifier
logreg = LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train, y_train)

# 4. Predict Matches on the full candidate set
predictions = logreg.predict(features)
final_matches = features.index[predictions]

print("Logistic Regression Classifier Trained and Applied.")
print(f"Final Matches found (Supervised Model): {len(final_matches)}")
print("-" * 30)

# -------------------------------------------------------------------
# E. Evaluation
# -------------------------------------------------------------------

precision_score = precision(true_matches, final_matches)
recall_score = recall(true_matches, final_matches)
f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) else 0

print("--- Evaluation (Supervised Model + SBERT) ---")
print(f"Precision: {precision_score:.4f} (Safety)")
print(f"Recall:    {recall_score:.4f} (Completeness)")
print(f"F1-Score:  {f1_score:.4f} (Overall Performance)")