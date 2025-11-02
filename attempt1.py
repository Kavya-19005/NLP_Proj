import recordlinkage
import pandas as pd
import numpy as np
from recordlinkage.datasets import load_febrl4
from recordlinkage import precision, recall
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # Added StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, make_scorer # Added make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx 

# --- A. Data Loading and Preprocessing ---

dfA, dfB, true_matches = load_febrl4(return_links=True) 
df_combined = pd.concat([dfA, dfB], axis=0) 

def create_contact_text(record):
    name = f"{record['given_name']} {record['surname']}".strip()
    id_field = str(record.get('soc_sec_id', '')).replace('.0', '')
    address = record.get('address_1', '')
    return f"{name} {id_field} {address}" 

dfA['contact_text'] = dfA.apply(create_contact_text, axis=1)
dfB['contact_text'] = dfB.apply(create_contact_text, axis=1)

# Note: Loading SBERT Model (This step can take a few seconds)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_A = model.encode(dfA['contact_text'].tolist(), convert_to_tensor=True, show_progress_bar=False) 
embeddings_B = model.encode(dfB['contact_text'].tolist(), convert_to_tensor=True, show_progress_bar=False) 

# --- B. Blocking & Feature Generation ---

indexer = recordlinkage.Index()
dfA['given_name_prefix'] = dfA['given_name'].str[:2]
dfB['given_name_prefix'] = dfB['given_name'].str[:2]
indexer.block(left_on='given_name_prefix', right_on='given_name_prefix')
indexer.block(left_on='surname', right_on='surname')
candidate_links = indexer.index(dfA, dfB) 

# --- C. Comparison (Feature Generation: Excluding highly selective features) ---

compare_cl = recordlinkage.Compare()
compare_cl.string('given_name', 'given_name', method='jarowinkler', label='name_jw')
compare_cl.string('surname', 'surname', method='jarowinkler', label='surname_jw')
# REMOVED: compare_cl.exact('date_of_birth', 'date_of_birth', label='dob_exact') 
# This highly selective feature is removed to force reliance on noisier text features (reducing overfitting suspicion).
compare_cl.string('address_1', 'address_1', method='damerau_levenshtein', threshold=0.8, label='address_dl')
features = compare_cl.compute(candidate_links, dfA, dfB)

# --- Add SBERT Similarity as a new Feature ---
sbert_scores = []
dfA_index_map = {idx: i for i, idx in enumerate(dfA.index)}
dfB_index_map = {idx: i for i, idx in enumerate(dfB.index)}

for id1, id2 in candidate_links:
    emb1 = embeddings_A[dfA_index_map[id1]]
    emb2 = embeddings_B[dfB_index_map[id2]]
    sim = util.cos_sim(emb1, emb2).item()
    sbert_scores.append(sim)

sbert_series = pd.Series(sbert_scores, index=candidate_links, name='sbert_sim')
features = pd.concat([features, sbert_series], axis=1)

# --- D. Classification and Match Identification ---

y_true = np.in1d(candidate_links, true_matches)

# Split data for robust evaluation (use a fixed sample size since dataset is larger)
sample_features, _, sample_y_true, _ = train_test_split(
    features, y_true, test_size=0.8, random_state=42, stratify=y_true
)
# Re-split sample features to get the X_train/X_test structure for the final metrics and visuals
X_train, X_test, y_train, y_test = train_test_split(
    sample_features, sample_y_true, test_size=0.3, random_state=42, stratify=sample_y_true
)

# 3. Train the Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)

# 4. Predict Matches on the full candidate set
predictions = model_rf.predict(features)
final_matches = features.index[predictions] 

# --- D.1 Model Scoring (Cross-Validation for Robustness) ---

# We will use cross-validation on the sample data for robust validation metrics.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'f1': make_scorer(f1_score), 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score)}

cv_results = cross_val_score(model_rf, sample_features, sample_y_true, cv=cv, scoring='f1', n_jobs=-1)

# To generate visuals, we still use the single X_test/y_test split (for demonstration)
y_pred_test = model_rf.predict(X_test)


print("\n--- Model Validation (Stratified 5-Fold CV) ---")
print(f"Average F1-Score: {np.mean(cv_results):.4f} (+/- {np.std(cv_results):.4f})")
print(f"Validation on Held-Out Test Set (Single Split):")
print(classification_report(y_test, y_pred_test, target_names=['Non-Match', 'Match']))

# --- E. Clustering and Golden Record Selection (Using NetworkX) ---

# 1. Clustering: Use NetworkX
G = nx.Graph()
G.add_edges_from(final_matches)
clusters = list(nx.connected_components(G))

# 2. Select Golden Record and Create Output
golden_records = []
for cluster_id, record_ids_set in enumerate(clusters):
    record_ids = list(record_ids_set)
    canonical_id = min(record_ids) 
    canonical_record_data = df_combined.loc[canonical_id].copy()
    canonical_record_data['canonical_id'] = cluster_id
    golden_records.append(canonical_record_data)

df_golden = pd.DataFrame(golden_records)

# --- F. Final Output and Visualizations ---

# 1. Print a summary (Added back the print statements for clarity)
print("\n--- Final Output Summary ---")
print(f"Total initial records (dfA + dfB): {len(df_combined):,}")
print(f"Total unique entities found (Golden Records): {len(df_golden):,}")

# 2. Calculate overall metrics for full evaluation
overall_precision = precision(true_matches, final_matches)
overall_recall = recall(true_matches, final_matches)
overall_f1 = f1_score(y_true, predictions) # Using sklearn's F1 on the full set

print(f"--- Overall Final Metrics (Full Dataset) ---")
print(f"F1-Score: {overall_f1:.4f}")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall: {overall_recall:.4f}")

# 3. Feature Importance Plot
feature_importances = pd.Series(model_rf.feature_importances_, index=features.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
# Using standard color map as the previous hue parameter caused deprecation warning
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
plt.title('Random Forest Feature Importance in Deduplication Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Generated Feature Importance chart: feature_importance.png")
# 

# 4. Confusion Matrix Plot (using the held-out test set predictions)
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Match', 'Match'], 
            yticklabels=['Non-Match', 'Match'])
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Generated Confusion Matrix chart: confusion_matrix.png")
# 

# 5. SBERT Score Distribution Plot (NEW VISUALIZATION)
# Get SBERT scores for True Matches and Non-Matches from the full candidate set
match_scores = features.loc[y_true, 'sbert_sim']
non_match_scores = features.loc[~y_true, 'sbert_sim'].sample(n=len(match_scores) * 2, random_state=42) # Sample non-matches for a balanced plot view

plt.figure(figsize=(10, 6))
sns.kdeplot(match_scores, label='True Matches', fill=True, alpha=0.6, linewidth=2)
sns.kdeplot(non_match_scores, label='Non-Matches (Sampled)', fill=True, alpha=0.6, linewidth=2)
plt.title('Distribution of SBERT Similarity Scores by Class')
plt.xlabel('SBERT Cosine Similarity Score')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('sbert_distribution.png')
print("Generated SBERT Score Distribution chart: sbert_distribution.png")

# 6. CSV Output
output_file = "deduplicated_contact_list.csv"
df_golden.to_csv(output_file, index=False)
print(f"Deduplication complete. Saved unique entities to: {output_file}")
