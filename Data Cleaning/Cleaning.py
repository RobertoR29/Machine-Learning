#!/usr/bin/env python3
import csv
import argparse
from collections import deque
import math

# Flow-level header
FLOW_HEADER = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
    "res_bdy_len", "payload_hex",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label"
]

# seconds of inactivity before we flush a flow
FLOW_TIMEOUT = 60.0  


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        # handle floats-in-string like "80.0"
        return int(float(v))
    except Exception:
        return default

#Make a canonical bidirectional flow key like 'A:port->B:port|PROTO'
def make_canonical_key(srcip, sport, dstip, dsport, proto):
    a = f"{srcip}:{sport}"
    b = f"{dstip}:{dsport}"
    proto = str(proto)
    if (a, b) <= (b, a):
        return f"{a}->{b}|{proto}", a, b
    else:
        return f"{b}->{a}|{proto}", b, a

#Create a new flow accumulator dict
def init_flow(srcip, sport, dstip, dsport, proto, t, payload_hex, state, service,
              swin, stcpb, is_ftp_login, ct_ftp_cmd, label):
    flow = {}

    # Origin direction
    flow["srcip"] = srcip
    flow["sport"] = sport
    flow["dstip"] = dstip
    flow["dsport"] = dsport
    flow["proto"] = proto

    flow["Stime"] = t
    flow["Ltime"] = t

    # bytes / pkts
    flow["sbytes"] = 0.0
    flow["dbytes"] = 0.0
    flow["Spkts"] = 0
    flow["Dpkts"] = 0

    # we’ll derive smeansz = sum / Spkts
    flow["smeansz_sum"] = 0.0  
    flow["dmeansz_sum"] = 0.0

    flow["sttl"] = 0
    flow["dttl"] = 0

    flow["sloss"] = 0
    flow["dloss"] = 0

    flow["service"] = service if service not in ("0", "", None) else "-"

    flow["swin"] = swin
    flow["dwin"] = 0
    flow["stcpb"] = stcpb
    flow["dtcpb"] = 0

    flow["trans_depth"] = 0
    flow["res_bdy_len"] = 0

    flow["payload_hex"] = payload_hex or ""

    # jitter / IAT stats 
    flow["Sf_last"] = None
    flow["Sf_count"] = 0
    flow["Sf_mean"] = 0.0
    flow["Sf_M2"] = 0.0

    flow["Db_last"] = None
    flow["Db_count"] = 0
    flow["Db_mean"] = 0.0
    flow["Db_M2"] = 0.0

    # TCP RTT pieces
    flow["syn_time"] = None
    flow["synack_time"] = None
    flow["ack_time"] = None

    # HTTP / FTP
    flow["ct_flw_http_mthd"] = 0
    flow["is_ftp_login"] = is_ftp_login
    flow["ct_ftp_cmd"] = ct_ftp_cmd

    flow["attack_cat"] = "0"
    flow["Label"] = label

    flow["state_seen"] = [] 

    flow["is_sm_ips_ports"] = int(srcip == dstip and sport == dsport)

    return flow

# Update inter-arrival/jitter stats for forward or backward dir
def update_iat(flow, t, forward=True):
    if forward:
        last = flow["Sf_last"]
        if last is not None:
            delta = t - last
            if delta < 0:
                delta = 0.0
            c = flow["Sf_count"]
            c_new = c + 1
            mean = flow["Sf_mean"]
            M2 = flow["Sf_M2"]

            delta2 = delta - mean
            mean_new = mean + delta2 / c_new
            M2_new = M2 + delta2 * (delta - mean_new)

            flow["Sf_count"] = c_new
            flow["Sf_mean"] = mean_new
            flow["Sf_M2"] = M2_new
        flow["Sf_last"] = t
    else:
        last = flow["Db_last"]
        if last is not None:
            delta = t - last
            if delta < 0:
                delta = 0.0
            c = flow["Db_count"]
            c_new = c + 1
            mean = flow["Db_mean"]
            M2 = flow["Db_M2"]

            delta2 = delta - mean
            mean_new = mean + delta2 / c_new
            M2_new = M2 + delta2 * (delta - mean_new)

            flow["Db_count"] = c_new
            flow["Db_mean"] = mean_new
            flow["Db_M2"] = M2_new
        flow["Db_last"] = t

# Approximate SYN / SYN-ACK / ACK timestamps from state bits
def update_tcp_handshake(flow, t, state_str, is_forward):
    if "TCP" not in str(flow["proto"]).upper():
        return

    # SYN fwd
    if is_forward and ("S" in state_str) and ("A" not in state_str):
        if flow["syn_time"] is None:
            flow["syn_time"] = t

    # SYN-ACK bwd
    if (not is_forward) and ("S" in state_str) and ("A" in state_str):
        if flow["synack_time"] is None:
            flow["synack_time"] = t

    # final ACK fwd
    if is_forward and ("A" in state_str) and ("S" not in state_str):
        if flow["ack_time"] is None:
            flow["ack_time"] = t

# Flush flows whose last-seen time is older than current_time - timeout
def flush_expired_flows(flows, expiry_queue, current_time, timeout, writer):
    threshold = current_time - timeout
    while expiry_queue and expiry_queue[0][0] <= threshold:
        ts, key = expiry_queue.popleft()
        flow = flows.get(key)
        if flow is None:
            continue
        # Only flush if this timestamp is the latest we have logged for this flow
        if flow["Ltime"] != ts:
            continue
        write_flow(flow, writer)
        del flows[key]

# Finalize derived fields and write flow to CSV
def write_flow(flow, writer):
    dur = max(0.0, flow["Ltime"] - flow["Stime"])
    flow["dur"] = dur

    # mean sizes
    if flow["Spkts"] > 0:
        flow["smeansz"] = flow["sbytes"] / flow["Spkts"]
    else:
        flow["smeansz"] = 0.0

    if flow["Dpkts"] > 0:
        flow["dmeansz"] = flow["dbytes"] / flow["Dpkts"]
    else:
        flow["dmeansz"] = 0.0

    # jitter & mean IAT
    if flow["Sf_count"] > 0:
        flow["Sintpkt"] = flow["Sf_mean"]
        flow["Sjit"] = math.sqrt(flow["Sf_M2"] / flow["Sf_count"])
    else:
        flow["Sintpkt"] = 0.0
        flow["Sjit"] = 0.0

    if flow["Db_count"] > 0:
        flow["Dintpkt"] = flow["Db_mean"]
        flow["Djit"] = math.sqrt(flow["Db_M2"] / flow["Db_count"])
    else:
        flow["Dintpkt"] = 0.0
        flow["Djit"] = 0.0

    # TCP RTT pieces
    syn = flow["syn_time"]
    synack = flow["synack_time"]
    ack = flow["ack_time"]
    synack_delay = 0.0
    ackdat_delay = 0.0
    if syn is not None and synack is not None:
        synack_delay = max(0.0, synack - syn)
    if synack is not None and ack is not None:
        ackdat_delay = max(0.0, ack - synack)
    flow["synack"] = synack_delay
    flow["ackdat"] = ackdat_delay
    flow["tcprtt"] = synack_delay + ackdat_delay

    # loads
    if dur > 0:
        flow["Sload"] = (flow["sbytes"] * 8.0) / dur
        flow["Dload"] = (flow["dbytes"] * 8.0) / dur
    else:
        flow["Sload"] = 0.0
        flow["Dload"] = 0.0

    # state: pick first seen or "-" if none
    if flow["state_seen"]:
        flow["state"] = flow["state_seen"][0]
    else:
        flow["state"] = "0"

    # ensure service is not empty
    if not flow["service"]:
        flow["service"] = "-"

    # ct_* = 0 in this streaming version
    flow["ct_state_ttl"] = 0
    flow["ct_srv_src"] = 0
    flow["ct_srv_dst"] = 0
    flow["ct_dst_ltm"] = 0
    flow["ct_src_ltm"] = 0
    flow["ct_src_dport_ltm"] = 0
    flow["ct_dst_sport_ltm"] = 0
    flow["ct_dst_src_ltm"] = 0

    # attack_cat/Label are already in flow 
    row = {
        "srcip": flow["srcip"],
        "sport": flow["sport"],
        "dstip": flow["dstip"],
        "dsport": flow["dsport"],
        "proto": flow["proto"],
        "state": flow["state"],
        "dur": flow["dur"],
        "sbytes": flow["sbytes"],
        "dbytes": flow["dbytes"],
        "sttl": flow["sttl"],
        "dttl": flow["dttl"],
        "sloss": flow["sloss"],
        "dloss": flow["dloss"],
        "service": flow["service"],
        "Sload": flow["Sload"],
        "Dload": flow["Dload"],
        "Spkts": flow["Spkts"],
        "Dpkts": flow["Dpkts"],
        "swin": flow["swin"],
        "dwin": flow["dwin"],
        "stcpb": flow["stcpb"],
        "dtcpb": flow["dtcpb"],
        "smeansz": flow["smeansz"],
        "dmeansz": flow["dmeansz"],
        "trans_depth": flow["trans_depth"],
        "res_bdy_len": flow["res_bdy_len"],
        "payload_hex": flow["payload_hex"],
        "Sjit": flow["Sjit"],
        "Djit": flow["Djit"],
        "Stime": flow["Stime"],
        "Ltime": flow["Ltime"],
        "Sintpkt": flow["Sintpkt"],
        "Dintpkt": flow["Dintpkt"],
        "tcprtt": flow["tcprtt"],
        "synack": flow["synack"],
        "ackdat": flow["ackdat"],
        "is_sm_ips_ports": flow["is_sm_ips_ports"],
        "ct_state_ttl": flow["ct_state_ttl"],
        "ct_flw_http_mthd": flow["ct_flw_http_mthd"],
        "is_ftp_login": flow["is_ftp_login"],
        "ct_ftp_cmd": flow["ct_ftp_cmd"],
        "ct_srv_src": flow["ct_srv_src"],
        "ct_srv_dst": flow["ct_srv_dst"],
        "ct_dst_ltm": flow["ct_dst_ltm"],
        "ct_src_ltm": flow["ct_src_ltm"],
        "ct_src_dport_ltm": flow["ct_src_dport_ltm"],
        "ct_dst_sport_ltm": flow["ct_dst_sport_ltm"],
        "ct_dst_src_ltm": flow["ct_dst_src_ltm"],
        "attack_cat": flow["attack_cat"],
        "Label": flow["Label"],
    }

    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Stream-clean packet-level CSV into UNSW-style flow-level CSV without loading everything into RAM."
    )
    parser.add_argument("input_csv", help="Huge packet-level CSV (uncleaned.csv)")
    parser.add_argument("output_csv", help="Output flow-level CSV (cleaned_flows.csv)")
    args = parser.parse_args()

    print(f"Streaming from: {args.input_csv}")
    print(f"Writing flows to: {args.output_csv}")

    # key -> flow accumulator
    flows = {}
    expiry_queue = deque() 

    with open(args.input_csv, "r", newline="") as fin, \
            open(args.output_csv, "w", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FLOW_HEADER)
        writer.writeheader()

        line_count = 0

        for row in reader:
            line_count += 1
            if line_count % 500000 == 0:
                print(f"Processed {line_count:,} packets... current active flows: {len(flows):,}")

            # parse needed fields from packet row
            srcip = row.get("srcip", "")
            dstip = row.get("dstip", "")
            sport = safe_int(row.get("sport", 0))
            dsport = safe_int(row.get("dsport", 0))
            proto = row.get("proto", "")

            t = safe_float(row.get("Stime", 0.0))
            pkt_len = safe_float(row.get("sbytes", 0.0))

            sttl_pkt = safe_int(row.get("sttl", 0))
            state = row.get("state", "0")
            service = row.get("service", "-")
            swin_pkt = safe_int(row.get("swin", 0))
            stcpb_pkt = safe_int(row.get("stcpb", 0))

            payload_hex = row.get("payload_hex", "")

            # HTTP/FTP hints from packet-level
            ct_flw_http_mthd_pkt = safe_int(row.get("ct_flw_http_mthd", 0))
            is_ftp_login_pkt = safe_int(row.get("is_ftp_login", 0))
            ct_ftp_cmd_pkt = safe_int(row.get("ct_ftp_cmd", 0))
            label_pkt = safe_int(row.get("Label", 0))

            # canonical key and direction
            key, a, b = make_canonical_key(srcip, sport, dstip, dsport, proto)
            is_forward = (f"{srcip}:{sport}" == a)

            if key not in flows:
                # new flow
                if is_forward:
                    f_srcip, f_sport, f_dstip, f_dsport = srcip, sport, dstip, dsport
                else:
                    # we still treat "a" as origin, but this packet is b->a
                    # origin srcip/sport = a side
                    a_ip, a_port = a.split(":")
                    b_ip, b_port = b.split(":")
                    f_srcip, f_sport = a_ip, safe_int(a_port)
                    f_dstip, f_dsport = b_ip, safe_int(b_port)

                flow = init_flow(
                    f_srcip, f_sport, f_dstip, f_dsport, proto, t,
                    payload_hex, state, service,
                    swin_pkt if is_forward else 0,
                    stcpb_pkt if is_forward else 0,
                    is_ftp_login_pkt, ct_ftp_cmd_pkt, label_pkt
                )
                flows[key] = flow
            else:
                flow = flows[key]

            # Update common timestamps
            if t < flow["Stime"]:
                flow["Stime"] = t
            if t > flow["Ltime"]:
                flow["Ltime"] = t

            # bytes / pkts, direction-aware
            if is_forward:
                flow["sbytes"] += pkt_len
                flow["Spkts"] += 1
                flow["smeansz_sum"] += pkt_len
                if flow["sttl"] == 0:
                    flow["sttl"] = sttl_pkt
                if flow["swin"] == 0 and swin_pkt != 0:
                    flow["swin"] = swin_pkt
                if flow["stcpb"] == 0 and stcpb_pkt != 0:
                    flow["stcpb"] = stcpb_pkt
                update_iat(flow, t, forward=True)
            else:
                flow["dbytes"] += pkt_len
                flow["Dpkts"] += 1
                flow["dmeansz_sum"] += pkt_len
                if flow["dttl"] == 0:
                    flow["dttl"] = sttl_pkt
                if flow["dwin"] == 0 and swin_pkt != 0:
                    flow["dwin"] = swin_pkt
                if flow["dtcpb"] == 0 and stcpb_pkt != 0:
                    flow["dtcpb"] = stcpb_pkt
                update_iat(flow, t, forward=False)

            # record state for later "mode"
            if state and state not in ("0", ""):
                flow["state_seen"].append(state)

            # service: keep first non-empty / non-"-"
            if flow["service"] in ("-", "", None) and service not in ("0", "", None):
                flow["service"] = service

            # payload: keep first non-empty
            if not flow["payload_hex"] and payload_hex:
                flow["payload_hex"] = payload_hex

            # HTTP / FTP aggregated
            if ct_flw_http_mthd_pkt > 0 and is_forward:
                flow["ct_flw_http_mthd"] += ct_flw_http_mthd_pkt
                flow["trans_depth"] += ct_flw_http_mthd_pkt

            if is_ftp_login_pkt > 0:
                flow["is_ftp_login"] = max(flow["is_ftp_login"], is_ftp_login_pkt)
            if ct_ftp_cmd_pkt > 0:
                flow["ct_ftp_cmd"] += ct_ftp_cmd_pkt

            # label aggregation
            if label_pkt == 1:
                flow["Label"] = 1

            # TCP handshake approximation
            update_tcp_handshake(flow, t, state, is_forward)

            # push into expiry queue
            expiry_queue.append((flow["Ltime"], key))
            # flush old flows
            flush_expired_flows(flows, expiry_queue, t, FLOW_TIMEOUT, writer)


        # Flush remaining flows
        print(f"Finished packets. Flushing remaining {len(flows):,} flows...")
        for key, flow in list(flows.items()):
            write_flow(flow, writer)
            del flows[key]

    print("Done streaming cleaning.")
    print(f"Output written to: {args.output_csv}")


if __name__ == "__main__":
    main()
