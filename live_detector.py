import threading
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from scapy.all import AsyncSniffer, IP, TCP, UDP


NSL_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]


class LiveFlowDetector:
    """
    Lightweight real-time network flow collector.

    Packets are aggregated into short flow windows and converted into
    an NSL-KDD-compatible feature frame.

    This is a telemetry bridge for the existing OracleShield detector;
    it does not retrain the classifier from live predictions.
    """

    def __init__(self, interface, window_seconds=5):
        self.interface = interface
        self.window_seconds = window_seconds

        self.running = False
        self.sniffer = None

        self.lock = threading.Lock()

        self.flows = defaultdict(self._new_flow)
        self.completed = []

        self.total_packets = 0
        self.total_bytes = 0

        self.started_at = None

    @staticmethod
    def _new_flow():
        return {
            "start": time.time(),
            "last": time.time(),
            "packets": 0,
            "bytes": 0,
            "src_bytes": 0,
            "dst_bytes": 0,
            "protocol": "other",
            "src": None,
            "dst": None,
            "src_port": None,
            "dst_port": None,
            "syn": 0,
            "rst": 0,
            "errors": 0,
        }

    def _packet_callback(self, pkt):
        if IP not in pkt:
            return

        ip = pkt[IP]

        src = ip.src
        dst = ip.dst

        proto = "other"
        sport = None
        dport = None
        flags = ""

        if TCP in pkt:
            proto = "tcp"
            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)
            flags = str(pkt[TCP].flags)

        elif UDP in pkt:
            proto = "udp"
            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)

        key = (
            src,
            dst,
            sport,
            dport,
            proto,
        )

        packet_len = len(pkt)

        with self.lock:
            flow = self.flows[key]

            now = time.time()

            flow["last"] = now
            flow["packets"] += 1
            flow["bytes"] += packet_len
            flow["src_bytes"] += packet_len
            flow["protocol"] = proto
            flow["src"] = src
            flow["dst"] = dst
            flow["src_port"] = sport
            flow["dst_port"] = dport

            if "S" in flags and "A" not in flags:
                flow["syn"] += 1

            if "R" in flags:
                flow["rst"] += 1

            self.total_packets += 1
            self.total_bytes += packet_len

    def _expire_flows(self):
        while self.running:

            now = time.time()
            expired = []

            with self.lock:
                for key, flow in list(self.flows.items()):

                    if now - flow["last"] >= self.window_seconds:

                        expired.append(flow)
                        del self.flows[key]

                for flow in expired:
                    self.completed.append(flow)

                self.completed = self.completed[-5000:]

            time.sleep(0.5)

    def start(self):
        if self.running:
            return

        self.running = True
        self.started_at = datetime.now()

        self.sniffer = AsyncSniffer(
            iface=self.interface,
            prn=self._packet_callback,
            store=False,
        )

        self.sniffer.start()

        threading.Thread(
            target=self._expire_flows,
            daemon=True,
        ).start()

    def stop(self):
        self.running = False

        if self.sniffer is not None:
            try:
                self.sniffer.stop()
            except Exception:
                pass

            self.sniffer = None

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "interface": self.interface,
                "packets": self.total_packets,
                "bytes": self.total_bytes,
                "active_flows": len(self.flows),
                "completed_flows": len(self.completed),
                "uptime": (
                    time.time() - self.started_at.timestamp()
                    if self.started_at
                    else 0
                ),
            }

    def get_recent_flows(self, n=50):
        with self.lock:
            flows = list(self.completed[-n:])

        return pd.DataFrame(
            [self._flow_to_nsl(f) for f in flows]
        )

    @staticmethod
    def _flow_to_nsl(flow):
        duration = max(
            0.001,
            flow["last"] - flow["start"]
        )

        packets = max(flow["packets"], 1)

        syn_rate = flow["syn"] / packets
        rst_rate = flow["rst"] / packets

        return {
            "duration": duration,

            "protocol_type": flow["protocol"],

            # Live service identification is intentionally conservative.
            # The trained model's service vocabulary is NSL-KDD-specific.
            "service": "other",

            "flag": "SF",

            "src_bytes": flow["src_bytes"],
            "dst_bytes": flow["dst_bytes"],

            "land": int(flow["src"] == flow["dst"]),
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,

            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,

            "count": packets,
            "srv_count": packets,

            "serror_rate": syn_rate,
            "srv_serror_rate": syn_rate,
            "rerror_rate": rst_rate,
            "srv_rerror_rate": rst_rate,

            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,

            "dst_host_count": packets,
            "dst_host_srv_count": packets,

            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,

            "dst_host_serror_rate": syn_rate,
            "dst_host_srv_serror_rate": syn_rate,
            "dst_host_rerror_rate": rst_rate,
            "dst_host_srv_rerror_rate": rst_rate,
        }
