# All imports required for Random Forest code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import ipaddress
from imblearn.over_sampling import SMOTE
import numpy as np

# 1. Load CSV
# Load clean and labeled csv 
df = pd.read_csv("labeled_cleaned_nopayload.csv")
print("Original dataset shape:", df.shape)

# 2. Clean attack_cat
# Remove spaces and replace missing or invalid categories with Benign
df["attack_cat"] = df["attack_cat"].astype(str).str.strip() # Remove spaces
df["attack_cat"] = df["attack_cat"].replace(["", " ", "nan", "None", "NONE", "NaN"], "Benign") # Fix empty/missing labels
df["attack_cat"] = df["attack_cat"].fillna("Benign") # Fill any remaining NaNs with Benign

print("Attack category counts after cleaning:\n", df["attack_cat"].value_counts())

# 3. Map attack categories to numeric labels
attack_mapping = {
    "Benign": 0,
    "Fuzzers": 1,
    "Analysis": 2,
    "Backdoors": 3,
    "DoS": 4,
    "Exploits": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9
}
# Map to numbers
df["attack_num"] = df["attack_cat"].map(attack_mapping).fillna(0).astype(int)

# 4. Downsample Benign
attack_df = df[df["attack_num"] != 0] # All attack samples
benign_df = df[df["attack_num"] == 0] # Benign samples

# Keep Benign to 50% of total attack samples
n_benign_keep = int(len(attack_df) * 0.5)
benign_df_downsampled = benign_df.sample(n=n_benign_keep, random_state=42)

df_balanced = pd.concat([attack_df, benign_df_downsampled], axis=0).sample(frac=1, random_state=42) # Combine and shuffle samples
print("Counts after Benign downsampling:")
print(df_balanced["attack_num"].value_counts())

# 5. Prepare features and target
X = df_balanced.drop(["attack_cat", "attack_num", "Label"], axis=1, errors="ignore") # Features
y = df_balanced["attack_num"] # Target

# Convert IP addresses to integers
def ip_to_int(ip):
    try:
        return int(ipaddress.IPv4Address(ip))
    except:
        return 0

for col in ["srcip", "dstip"]:
    if col in X.columns:
        X[col] = X[col].apply(ip_to_int)

# Convert remaining categorical/text columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X.astype(np.float32))

# 6. SMOTE Resampling
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print("Shape after SMOTE:", X_resampled.shape, y_resampled.shape)

# 7. Train/Test Split
# 80/20 training/testing split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 8. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 9. Make predictions on Test Set
y_pred = rf.predict(X_test)

# 10. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

labels_present = sorted(np.unique(y_test))
target_names_full = list(attack_mapping.keys())
target_names_filtered = [name for i, name in enumerate(target_names_full) if i in labels_present]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_filtered, zero_division=0))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=labels_present)

plt.figure(figsize=(12, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.colorbar()

tick_marks = range(len(labels_present))
plt.xticks(tick_marks, target_names_filtered, rotation=45, ha="right")
plt.yticks(tick_marks, target_names_filtered)

# Add numbers to plot
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
