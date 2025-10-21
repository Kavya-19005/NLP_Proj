# This script performs cross-dataset record linkage on FEBRL 4 (dfA and dfB) using traditional string metrics
# and SBERT embeddings. Candidate pairs are generated via compound blocking on name prefixes and surnames. 
# A Random Forest classifier predicts matches on the full set, and performance is evaluated with precision, 
# recall, and F1-score.

import recordlinkage
import pandas as pd
import numpy as np
from recordlinkage.datasets import load_febrl4 # Loading FEBRL 4
from recordlinkage import precision, recall
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- A. Data Loading and Preparation (FEBRL 4) ---

# Load FEBRL 4: Returns two dataframes (dfA, dfB) and the true links
# dfA and dfB each have 5000 records, total 10,000 records
dfA, dfB, true_matches = load_febrl4(return_links=True) 

print(f"Total Records (dfA + dfB): {len(dfA) + len(dfB)}")
print(f"True Match Pairs: {len(true_matches)}") # Should be 5000 links
print("-" * 30)

# --- 1. Prepare Text for SBERT on BOTH DataFrames ---

def create_contact_text(record):
    """Combines FEBRL fields into a single text string for SBERT encoding."""
    # Using a simple concatenation format for optimal SBERT performance
    name = f"{record['given_name']} {record['surname']}".strip()
    id_field = str(record.get('soc_sec_id', '')).replace('.0', '')
    address = record.get('address_1', '')
    return f"{name} {id_field} {address}" 

dfA['contact_text'] = dfA.apply(create_contact_text, axis=1)
dfB['contact_text'] = dfB.apply(create_contact_text, axis=1)


# --- 2. Encode all records using Sentence-Transformers ---
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding records with SBERT...")

# Encode both dataframes separately
embeddings_A = model.encode(dfA['contact_text'].tolist(), convert_to_tensor=True, show_progress_bar=False) 
embeddings_B = model.encode(dfB['contact_text'].tolist(), convert_to_tensor=True, show_progress_bar=False) 

print("Encoding complete.")
print("-" * 30)

# -------------------------------------------------------------------
# B. Blocking (Candidate Selection) - Record Linkage Style
# -------------------------------------------------------------------

indexer = recordlinkage.Index()

# Create prefix columns for compound blocking on BOTH dataframes
dfA['given_name_prefix'] = dfA['given_name'].str[:2]
dfB['given_name_prefix'] = dfB['given_name'].str[:2]

# Compound Blocking between dfA and dfB (linking two files)
indexer.block(left_on='given_name_prefix', right_on='given_name_prefix') # Block on prefix
indexer.block(left_on='surname', right_on='surname')                   # Block on full surname

# The indexer.index call now takes both dataframes
candidate_links = indexer.index(dfA, dfB) 

print(f"Candidate Pairs after Compound Blocking: {len(candidate_links):,}")
print("-" * 30)

# -------------------------------------------------------------------
# C. Comparison (Feature Generation: String Metrics + SBERT)
# -------------------------------------------------------------------

compare_cl = recordlinkage.Compare()

# 1. Traditional String Metrics (applied between dfA and dfB)
compare_cl.string('given_name', 'given_name', method='jarowinkler', label='name_jw')
compare_cl.string('surname', 'surname', method='jarowinkler', label='surname_jw')
compare_cl.exact('date_of_birth', 'date_of_birth', label='dob_exact')
compare_cl.string('address_1', 'address_1', method='damerau_levenshtein', threshold=0.8, label='address_dl')

# Compute the features for traditional metrics
features = compare_cl.compute(candidate_links, dfA, dfB) # Note: takes both dfA and dfB

# --- Add SBERT Similarity as a new Feature ---
sbert_scores = []
# Create index maps for faster embedding lookup
dfA_index_map = {idx: i for i, idx in enumerate(dfA.index)}
dfB_index_map = {idx: i for i, idx in enumerate(dfB.index)}

for id1, id2 in candidate_links:
    # id1 is from dfA, id2 is from dfB
    emb1 = embeddings_A[dfA_index_map[id1]]
    emb2 = embeddings_B[dfB_index_map[id2]]
    
    sim = util.cos_sim(emb1, emb2).item()
    sbert_scores.append(sim)

sbert_series = pd.Series(sbert_scores, index=candidate_links, name='sbert_sim')
features = pd.concat([features, sbert_series], axis=1)

print(f"Features created for {len(features)} candidate pairs.")
print("-" * 30)

# -------------------------------------------------------------------
# D. Classification (Random Forest)
# -------------------------------------------------------------------

# 1. Create Training Labels (y) - The robust alignment fix
y_true = np.in1d(candidate_links, true_matches)

# 2. Split data for robust evaluation (use a fixed sample size since dataset is larger)
# We will use a smaller sample (e.g., 20% of candidates) for train/test split speed
sample_features, _, sample_y_true, _ = train_test_split(
    features, y_true, test_size=0.8, random_state=42, stratify=y_true
)

# Use the full feature set for the final prediction
X_train, X_test, y_train, y_test = train_test_split(
    sample_features, sample_y_true, test_size=0.3, random_state=42, stratify=sample_y_true
)

# 3. Train the Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)

# 4. Predict Matches on the full candidate set
predictions = model_rf.predict(features)
final_matches = features.index[predictions]

print("Random Forest Classifier (FEBRL 4) Trained and Applied.")
print(f"Final Matches found: {len(final_matches)}")
print("-" * 30)

# -------------------------------------------------------------------
# E. Evaluation
# -------------------------------------------------------------------

precision_score = precision(true_matches, final_matches)
recall_score = recall(true_matches, final_matches)
f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) else 0

print("--- Evaluation (FEBRL 4: Random Forest + SBERT) ---")
print(f"Precision (Safety): {precision_score:.4f}")
print(f"Recall (Completeness):    {recall_score:.4f}")
print(f"F1-Score (Overall Performance):  {f1_score:.4f}")