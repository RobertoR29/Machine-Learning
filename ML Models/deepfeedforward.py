import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import random
import os


# Setting the seed 

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Specifies the device being using in the classification ("ie cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Loading the Data

csv_path = "/Users/ashley/Downloads/labeled_cleaned_nopayload.csv"
if not os.path.exists(csv_path):
    raise SystemExit(f"Data file not found: {csv_path}")

df = pd.read_csv(csv_path)
print("Loaded dataset shape:", df.shape)


# This section of code will take the attack names that were given and corrolate them to a number 0-9

df["attack_cat"] = df["attack_cat"].astype(str).str.strip().str.title()
df["attack_cat"] = df["attack_cat"].replace({
    "Backdoor": "Backdoors",
    "Backdoors.": "Backdoors",
    "Dos": "Dos",
    "Ddos": "Dos",
    "Exploit": "Exploits",
    "Shell Code": "Shellcode",
    "Reconnaissance.": "Reconnaissance",
    "Nan": "Benign",
    "None": "Benign"
})

attack_mapping = {
    "Benign": 0,
    "Fuzzers": 1,
    "Analysis": 2,
    "Backdoors": 3,
    "Dos": 4,
    "Exploits": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9
}

df["Label"] = df["attack_cat"].map(attack_mapping)

# Check unmapped
if df["Label"].isna().sum() > 0:
    print(df[df["Label"].isna()]["attack_cat"].unique())
    raise SystemExit("Fix unmapped categories.")

print("Label counts before sampling:")
print(df["Label"].value_counts().sort_index())


# converts each string value into an integer

to_encode = [c for c in df.columns if df[c].dtype == "object" and c not in ("attack_cat", "Label")]
for col in to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# This section reduces the amount of beingn samples within the processed dataset to make the data more even 

attack_df = df[df["Label"] != 0]
benign_df = df[df["Label"] == 0]

n_benign_keep = int(len(attack_df) * 0.5)
benign_df_down = benign_df.sample(n=n_benign_keep, random_state=seed)

df_bal = pd.concat([attack_df, benign_df_down], axis=0).sample(frac=1, random_state=seed)

print("Label counts after balancing:")
print(df_bal["Label"].value_counts().sort_index())


y = df_bal["Label"].astype(int).values
X_df = df_bal.drop(columns=["attack_cat", "Label"]).apply(
    pd.to_numeric, errors="coerce"
).fillna(0.0)

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values.astype(np.float32))


# SMOTE

sm = SMOTE(random_state=seed)
X_res, y_res = sm.fit_resample(X, y)


# This section does the Train / Validation / Test Split

# First split off test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_res, y_res, test_size=0.20, random_state=seed, stratify=y_res
)

# Then split remaining 80% into 60/20 (train/val)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
)


print("Shapes:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)


# This section creates the Model with the hidden layers and units

class DFFN(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = DFFN(input_dim=X.shape[1]).to(device)
print(model)

# This section does will train the model on the 60% train split 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # This will compute the validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            out = model(vx)
            _, pred = torch.max(out, 1)
            total += vy.size(0)
            correct += (pred == vy).sum().item()

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")


# This section is the test Evaluation

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        out = model(tx)
        _, pred = torch.max(out, 1)
        all_preds.append(pred.cpu().numpy())
        all_true.append(ty.numpy())

all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)

cm = confusion_matrix(all_true, all_preds)

print("\nClassification Report:")
target_names = [
    "Benign", "Fuzzers", "Analysis", "Backdoors", "DoS",
    "Exploits", "Generic", "Recon", "Shellcode", "Worms"
]

unique_labels = sorted(np.unique(all_true))

#This will print out the classification report that shows the precision, recall, f1-score, and support for all 10 classes

print(classification_report(
    all_true,
    all_preds,
    labels=unique_labels,             
    target_names=[target_names[i] for i in unique_labels],
    digits=4
))

print("Macro F1:", f1_score(all_true, all_preds, average="macro"))

#This creates an orange Confusion Matrix Plot to show how well the model performed visually 

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
