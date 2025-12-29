import pandas as pd
import glob

CLEANED = "cleaned2.csv"
GROUND_FOLDER = "GroundTruth" # folder containing UNSW truth files
OUTPUT_SAFE = "labeled_cleaned_nopayload.csv"


# Load Cleaned Flow CSV
print("Loading cleaned CSV:", CLEANED)
df = pd.read_csv(CLEANED)

df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")
df["Stime"] = pd.to_numeric(df["Stime"], errors="coerce")

# Create matching timestamp
df["ts_sec"] = df["Stime"].astype(float).round().astype(int)

#Load and combine all 4 grounf truth files
gt_files = glob.glob(f"{GROUND_FOLDER}/*.csv")
gt_list = []

for file in gt_files:
    print("Loading ground truth:", file)
    gt = pd.read_csv(file, header=None)

    # Assign UNSW column names
    gt.columns = [
        "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes",
        "sttl","dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts",
        "swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth",
        "res_bdy_len","Sjit","Djit","start","end","Sintpkt","Dintpkt","tcprtt",
        "synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
        "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm",
        "ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm",
        "attack_cat","Label"
    ]

    gt_list.append(gt)

# Combine all four truth files
gt_all = pd.concat(gt_list, ignore_index=True)
print("Combined ground truth rows:", len(gt_all))

# Type cleanup
gt_all["sport"] = pd.to_numeric(gt_all["sport"], errors="coerce")
gt_all["dsport"] = pd.to_numeric(gt_all["dsport"], errors="coerce")
gt_all["start"] = pd.to_numeric(gt_all["start"], errors="coerce")

# Add time column for merging
gt_all["ts_sec"] = gt_all["start"].astype(int)

# Create reserved ground truth for bidirectional match
print("Creating reversed-direction ground truth...")

gt_rev = gt_all.copy()

gt_rev = gt_rev.rename(columns={
    "srcip": "dstip",
    "sport": "dsport",
    "dstip": "srcip",
    "dsport": "sport",
})

# Combine forward and reverse GT
gt_bi = pd.concat([gt_all, gt_rev], ignore_index=True)
print("Bi-directional ground truth rows:", len(gt_bi))

# Default Labeling Before Matching
df["attack_cat"] = "Benign"
df["Label"] = 0


# Perform Bidirectional Merge
merge_cols = ["srcip","sport","dstip","dsport","ts_sec"]

print("Performing bidirectional merge...")

merged = df.merge(
    gt_bi[merge_cols + ["attack_cat", "Label"]],
    how="left",
    on=merge_cols,
    suffixes=("", "_gt")
)

# If GT label exists
merged["attack_cat"] = merged["attack_cat_gt"].fillna(merged["attack_cat"])
merged["Label"] = merged["Label_gt"].fillna(0)

# Cleanup
merged = merged.drop(columns=["attack_cat_gt", "Label_gt", "ts_sec"])

# Remove Payload
if "payload_hex" in merged.columns:
    print("Removing payload column for safety...")
    merged = merged.drop(columns=["payload_hex"])

# Save Output
merged.to_csv(OUTPUT_SAFE, index=False)
print("Saved labeled file:", OUTPUT_SAFE)
