#!/usr/bin/env python3
import sys
import csv
import argparse
import os
from scapy.all import PcapReader, IP, TCP, UDP, ICMP, Raw
from tqdm import tqdm

# Map common ports to service names
PORT_SERVICE_MAP = {
    80: "http", 8080: "http", 443: "https", 21: "ftp", 20: "ftp-data",
    22: "ssh", 25: "smtp", 53: "dns", 110: "pop3", 143: "imap"
}

# Strings used to detect HTTP request methods
HTTP_METHODS = ("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ")

# CSV header fields 
HEADER = [
    "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes",
    "sttl","dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts",
    "swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len",
    "payload_hex","Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt",
    "tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
    "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm",
    "ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm",
    "attack_cat","Label"
]

# Infer a service name based on common port numbers
def infer_service_from_ports(sport, dport):
    if dport in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[dport]
    if sport in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[sport]
    return "-"

# Convert TCP flags to a readable value
def tcp_flags_to_state(tcp_layer):
    try:
        return str(tcp_layer.flags)
    except:
        return "-"

#Extracts features from a single packet and returns a row dict
def parse_packet(pkt):
    # Initialize row with default values
    row = {k: 0 for k in HEADER}
    # Timestamp fields
    ts = getattr(pkt, "time", 0)
    row["Stime"] = ts
    row["Ltime"] = ts

    # Payload extraction
    raw_bytes = bytes(pkt[Raw]) if Raw in pkt else b""
    row["payload_hex"] = raw_bytes.hex()
    # Basic size and packet count fields
    pkt_len = len(pkt)
    row["sbytes"] = pkt_len
    row["Spkts"] = 1
    row["smeansz"] = float(pkt_len)
    
    # Skip non-IP packets
    if IP not in pkt:
        row["proto"] = "NONIP"
        return row
    
    # IP layer fields
    ip = pkt[IP]
    row["srcip"] = ip.src
    row["dstip"] = ip.dst
    row["sttl"] = ip.ttl
    
    # Handle TCP packets
    if TCP in pkt:
        tcp = pkt[TCP]
        row["proto"] = "TCP"
        row["sport"] = tcp.sport
        row["dsport"] = tcp.dport
        row["state"] = tcp_flags_to_state(tcp)
        row["swin"] = tcp.window
        row["stcpb"] = tcp.seq
        # Decode payload to detect HTTP/FTP
        text = raw_bytes.decode("utf-8", errors="ignore")
        # True if HTTP method detected
        row["ct_flw_http_mthd"] = int(any(text.startswith(m) for m in HTTP_METHODS))
        # Detect FTP login attempts
        row["is_ftp_login"] = int("USER " in text.upper() or "PASS " in text.upper())
        row["ct_ftp_cmd"] = row["is_ftp_login"]
        row["res_bdy_len"] = len(raw_bytes)
        
    # Handle UDP packets
    elif UDP in pkt:
        udp = pkt[UDP]
        row["proto"] = "UDP"
        row["sport"] = udp.sport
        row["dsport"] = udp.dport
        row["res_bdy_len"] = len(raw_bytes)
        
    # Handle ICMP packets
    elif ICMP in pkt:
        row["proto"] = "ICMP"
        row["res_bdy_len"] = 0
        
    # Other IP protocols
    else:
        row["proto"] = str(ip.proto)
        row["res_bdy_len"] = len(raw_bytes)
        
    # Infer application protocol
    row["service"] = infer_service_from_ports(row["sport"], row["dsport"])
    # Check if source and destination IP/ports match exactly
    row["is_sm_ips_ports"] = int(row["srcip"] == row["dstip"] and row["sport"] == row["dsport"])
    # Approximate TTL
    ttl = row["sttl"]
    
    if ttl > 0:
        row["ct_state_ttl"] = (
            1 if ttl < 32 else 2 if ttl < 64 else 3 if ttl < 128 else 4
        )

    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="PCAP file or folder of PCAPs")
    parser.add_argument("output_csv", help="Output CSV file")
    args = parser.parse_args()

    # Build a list of PCAP files
    if os.path.isdir(args.input_path):
        pcaps = sorted([
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith((".pcap", ".pcapng"))
        ])
    else:
        pcaps = [args.input_path]

    if not pcaps:
        print("No PCAPs found.")
        return

    print(f"Processing {len(pcaps)} PCAP files...")

    # Open output CSV once, append everything
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        #Process each PCAP
        for p in pcaps:
            print(f"\nReading {p}...")
            try:
                with PcapReader(p) as pcap:
                    for pkt in tqdm(pcap, desc=os.path.basename(p)):
                        row = parse_packet(pkt)
                        writer.writerow(row)

            except Exception as e:
                print(f"Error reading {p}: {e}")

    print(f"\nDone! Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
