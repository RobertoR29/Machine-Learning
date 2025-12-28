#!/usr/bin/env python3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# seed created to ensure reproducibility while testing for best parameters
seed = 42


# Load Dataset, change to your dataset path, we have it in the same folder
csv_path = "labeled_cleaned_nopayload.csv"
df = pd.read_csv(csv_path)

# Select Attack Column
attack_col = "attack_cat"
print(f"Using attack category column: {attack_col}")

print("\nClass distribution BEFORE encoding:")
print(df[attack_col].value_counts())


# Encode attack labels (keep for multi-class reporting)
# Makes it so that each unique string label is mapped to an integer for simplicity for the model
label_encoder = LabelEncoder()
df["attack_cat_enc"] = label_encoder.fit_transform(df[attack_col])

print("\nEncoded class mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i}: {cls}")

# Makes labels  sequential 0, 1, 2, all the way to N-1 (this keeps mapping consistent)
unique_labels = sorted(df["attack_cat_enc"].unique())
label_map = {old: new for new, old in enumerate(unique_labels)}
df["attack_cat_enc"] = df["attack_cat_enc"].map(label_map)

print("\nRemapped classes:")
print(sorted(df["attack_cat_enc"].unique()))


# starting data balancing, starting with making benign less frequent, 50% in our training.testing
print("\nApplying benign down-sampling...")

# attack_df contains all non-benign samples, 1-8
attack_df = df[df["Label"] != 0]
benign_df = df[df["Label"] == 0]

#  keep benign samples to 50% of attack samples
n_benign_keep = int(len(attack_df) * 0.5)
benign_df_downsampled = benign_df.sample(n=n_benign_keep, random_state=seed)

# put them back together and then randomize
df_balanced = pd.concat([attack_df, benign_df_downsampled], axis=0).sample(frac=1, random_state=seed)

print("Label value counts after Benign downsampling:")
print(df_balanced["Label"].value_counts().sort_index())


# encode the rest of the attack categories to numbers
print("\nPreparing features and multi-class target (attack_cat_enc)...")

# target is the multi class encoding created earlier
y = df_balanced["attack_cat_enc"].astype(int).values

# drop string columns (attack_cat) and Label (used only for downsampling)
# keep attack_cat_enc (target)
X_df = df_balanced.drop(columns=[attack_col, "Label"], errors="ignore").copy()
# drop the target column from features if present
if "attack_cat_enc" in X_df.columns:
    X_df = X_df.drop(columns=["attack_cat_enc"])

# change to numeric and for ones that weren't labeled, set to 0.0
X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values.astype(np.float32))


# SMOTE oversampling (multiclass)
print("\nApplying SMOTE (multiclass)...")

sm = SMOTE(random_state=seed)
X_resampled, y_resampled = sm.fit_resample(X, y)

print("Shape after SMOTE resampling:", X_resampled.shape, y_resampled.shape)


# Train/Test split, set to .8 and .2 respectively, xgboost doesn't "remember"
# training so no need for a validation set here
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=seed, stratify=y_resampled
)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")


# Model Training, these were best results, increasing 
# max depth further increased overall accuracy very slightly, but overall should be kept low
model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False
)

print("\nTraining model...")
model.fit(X_train, y_train)


# Evaluation, printing overall accuracy and per class statistics
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# make confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
# how the confusion matrix shows
sns.heatmap(
    cm,
    annot=True,           # show numbers
    fmt='d',              # integer format
    cmap='Oranges',       # color map
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()