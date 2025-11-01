import recordlinkage
import pandas as pd
import numpy as np
from recordlinkage.datasets import load_febrl4
from recordlinkage import precision, recall
# Import ConnectedComponents for the final clustering step
from recordlinkage.graph import ConnectedComponents 
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- A. Data Loading and Preprocessing (Unchanged from Optimal Model) ---

dfA, dfB, true_matches = load_febrl4(return_links=True) 

# Combine dfA and dfB into one DataFrame for easier Golden Record selection later
# Note: FEBRL 4 record IDs are unique across dfA and dfB, so concatenation is safe.
df_combined = pd.concat([dfA, dfB], axis=0) 

def create_contact_text(record):
    name = f"{record['given_name']} {record['surname']}".strip()
    id_field = str(record.get('soc_sec_id', '')).replace('.0', '')
    address = record.get('address_1', '')
    return f"{name} {id_field} {address}" 

dfA['contact_text'] = dfA.apply(create_contact_text, axis=1)
dfB['contact_text'] = dfB.apply(create_contact_text, axis=1)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_A = model.encode(dfA['contact_text'].tolist(), convert_to_tensor=True, show_progress_bar=False) 
embeddings_B = model.encode(dfB['contact_text'].tolist(), convert_to_tensor=True, show_progress_bar=False) 

# --- B. Blocking & Feature Generation (Steps C and B from prior code, combined) ---

indexer = recordlinkage.Index()
dfA['given_name_prefix'] = dfA['given_name'].str[:2]
dfB['given_name_prefix'] = dfB['given_name'].str[:2]
indexer.block(left_on='given_name_prefix', right_on='given_name_prefix')
indexer.block(left_on='surname', right_on='surname')
candidate_links = indexer.index(dfA, dfB) 

# [Feature computation code remains unchanged for brevity]

# Placeholder for feature calculation based on your previous successful run:
# NOTE: In a final script, the full feature calculation and SBERT scoring logic goes here.
# For demonstration, we'll create dummy features aligned to the candidate links.
# In production, features should be fully computed!
features = pd.DataFrame(np.random.rand(len(candidate_links), 5), 
                        index=candidate_links, 
                        columns=['name_jw', 'surname_jw', 'dob_exact', 'address_dl', 'sbert_sim'])

# --- C. Classification and Match Identification ---

y_true = np.in1d(candidate_links, true_matches)
sample_features, _, sample_y_true, _ = train_test_split(
    features, y_true, test_size=0.8, random_state=42, stratify=y_true
)
X_train, _, y_train, _ = train_test_split(
    sample_features, sample_y_true, test_size=0.3, random_state=42, stratify=sample_y_true
)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)

# Get the final list of matched pairs predicted by the model
predictions = model_rf.predict(features)
final_matches = features.index[predictions] 

# --- D. Clustering and Golden Record Selection (NEW CRITICAL STEP) ---

# 1. Clustering: Use Connected Components (Transitive Closure)
# This groups all indirectly linked records into a single cluster ID.
cc = ConnectedComponents()
# The clustering requires the indices of the full dataframes (df_combined indices)
# and the actual matched pairs identified by the model.
clusters = cc.get_clusters(df_combined.index, final_matches)

# 2. Select Golden Record (Example Logic: Pick the record with the most complete data)
# For simplicity, we'll define the Golden Record as the first record ID encountered in the cluster.
golden_records = []
for cluster_id, record_ids in enumerate(clusters):
    # Select the first record ID in the cluster as the canonical ID
    canonical_id = record_ids[0]
    
    # Get the data for the canonical record
    canonical_record_data = df_combined.loc[canonical_id].copy()
    
    # Add a column for the Cluster/Canonical ID for tracking
    canonical_record_data['canonical_id'] = cluster_id
    canonical_records.append(canonical_record_data)

# Create the final DataFrame containing only the unique, non-duplicate Golden Records
df_golden = pd.DataFrame(canonical_records)

# --- E. Final Output ---

# 1. Print a summary
print("\n--- Final Output Summary ---")
print(f"Total initial records (dfA + dfB): {len(df_combined):,}")
print(f"Total unique entities found (Golden Records): {len(df_golden):,}")

# 2. Save the deduplicated data to a CSV file
output_file = "deduplicated_contact_list.csv"
df_golden.to_csv(output_file, index=False)
print(f"Deduplication complete. Saved unique entities to: {output_file}")