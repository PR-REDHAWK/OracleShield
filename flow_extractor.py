"""OracleShield Flow Extractor & Deep Temporal Telemetry Engine.

Extracts bidirectional network flow metrics, inter-arrival timing statistics,
TCP flag dynamics, and service diversity to encode 16-D temporal network states.
Supports live streaming packets, synthetic traffic generators, and PCAP ingestion.
"""

import time
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any

try:
    import scapy.all as scapy
    HAS_SCAPY = True
except Exception:
    HAS_SCAPY = False


SERVICE_PORTS = {
    80: 'http', 443: 'https', 22: 'ssh', 21: 'ftp', 25: 'smtp',
    53: 'dns', 3306: 'mysql', 8080: 'http_alt', 23: 'telnet'
}


class PacketRecord:
    """Represents a single parsed network packet with deep header telemetry."""

    def __init__(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str = 'TCP',
        payload_bytes: int = 0,
        tcp_flags: Optional[Dict[str, bool]] = None,
        ttl: int = 64,
        timestamp: Optional[float] = None
    ):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol.upper()
        self.payload_bytes = payload_bytes
        self.tcp_flags = tcp_flags or {'SYN': False, 'FIN': False, 'RST': False, 'ACK': True, 'PSH': False}
        self.ttl = ttl
        self.timestamp = timestamp or time.time()

    @property
    def flow_key(self) -> Tuple[str, int, str, int, str]:
        """Bidirectional 5-tuple key for flow grouping."""
        if (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port):
            return (self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.protocol)
        return (self.dst_ip, self.dst_port, self.src_ip, self.src_port, self.protocol)


class FlowTracker:
    """Tracks and computes temporal dynamics for a single network flow."""

    def __init__(self, flow_key: Tuple[str, int, str, int, str]):
        self.flow_key = flow_key
        self.packets: List[PacketRecord] = []
        self.start_time = None
        self.last_time = None
        self.src_bytes = 0
        self.dst_bytes = 0
        self.syn_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.service = SERVICE_PORTS.get(flow_key[1], SERVICE_PORTS.get(flow_key[3], 'other'))

    def add_packet(self, packet: PacketRecord):
        if not self.packets:
            self.start_time = packet.timestamp

        self.packets.append(packet)
        self.last_time = packet.timestamp

        # Directional byte count
        if packet.src_ip == self.flow_key[0] and packet.src_port == self.flow_key[1]:
            self.src_bytes += packet.payload_bytes
        else:
            self.dst_bytes += packet.payload_bytes

        # Flag telemetry
        if packet.tcp_flags.get('SYN'):
            self.syn_count += 1
        if packet.tcp_flags.get('FIN'):
            self.fin_count += 1
        if packet.tcp_flags.get('RST'):
            self.rst_count += 1

    @property
    def duration(self) -> float:
        if self.start_time and self.last_time:
            return max(0.0, self.last_time - self.start_time)
        return 0.0

    @property
    def inter_arrival_stats(self) -> Tuple[float, float]:
        """Computes mean and std of packet inter-arrival time (seconds)."""
        if len(self.packets) < 2:
            return 0.0, 0.0
        diffs = [
            self.packets[i].timestamp - self.packets[i - 1].timestamp
            for i in range(1, len(self.packets))
        ]
        return float(np.mean(diffs)), float(np.std(diffs))

    def to_dict(self) -> Dict[str, Any]:
        mean_iat, std_iat = self.inter_arrival_stats
        total_pkt = len(self.packets)
        return {
            'src_ip': self.flow_key[0],
            'src_port': self.flow_key[1],
            'dst_ip': self.flow_key[2],
            'dst_port': self.flow_key[3],
            'protocol': self.flow_key[4],
            'service': self.service,
            'duration': self.duration,
            'src_bytes': self.src_bytes,
            'dst_bytes': self.dst_bytes,
            'packet_count': total_pkt,
            'inter_arrival_mean': mean_iat,
            'inter_arrival_std': std_iat,
            'syn_rate': self.syn_count / max(1, total_pkt),
            'rst_rate': self.rst_count / max(1, total_pkt),
            'fin_rate': self.fin_count / max(1, total_pkt)
        }


class FlowStateEncoder:
    """Encodes active flow windows into 16-D temporal network state vectors."""

    @staticmethod
    def encode_flows_to_state(flows: List[FlowTracker], window_size: int = 50) -> np.ndarray:
        """Converts flow records into a normalized 16-D network state vector S_t."""
        if not flows:
            return np.zeros(16, dtype=np.float32)

        flow_dicts = [f.to_dict() for f in flows[-window_size:]]
        df = pd.DataFrame(flow_dicts)

        n_flows = len(df)
        src_bytes = df['src_bytes'].values
        dst_bytes = df['dst_bytes'].values
        durations = df['duration'].values
        syn_rates = df['syn_rate'].values
        rst_rates = df['rst_rate'].values
        services = df['service'].values

        # Heuristic attack category detection from deep flow telemetry
        dos_count = np.sum((durations < 1.0) & (syn_rates > 0.5) & (src_bytes < 200))
        probe_count = np.sum((rst_rates > 0.4) | (df['dst_port'].nunique() > n_flows * 0.4))
        r2l_count = np.sum((df['service'].isin(['ssh', 'ftp', 'telnet'])) & (src_bytes > 500) & (durations > 2.0))
        u2r_count = np.sum((dst_bytes > 50000) & (src_bytes > 20000))

        total_attacks = dos_count + probe_count + r2l_count + u2r_count
        attack_rate = min(1.0, float(total_attacks / max(1, n_flows)))

        dos_rate = float(dos_count / max(1, n_flows))
        probe_rate = float(probe_count / max(1, n_flows))
        r2l_rate = float(r2l_count / max(1, n_flows))
        u2r_rate = float(u2r_count / max(1, n_flows))

        mean_src_bytes = float(np.mean(src_bytes))
        mean_dst_bytes = float(np.mean(dst_bytes))
        mean_duration = float(np.mean(durations))
        mean_count = float(n_flows)
        mean_srv_count = float(df['service'].value_counts().mean()) if 'service' in df else 1.0
        mean_serror_rate = float(np.mean(syn_rates))
        mean_rerror_rate = float(np.mean(rst_rates))

        unique_services = df['service'].nunique() if 'service' in df else 1
        service_counts = df['service'].value_counts(normalize=True) if 'service' in df else pd.Series([1.0])
        same_srv_rate = float(service_counts.iloc[0]) if len(service_counts) > 0 else 1.0
        diff_srv_rate = float(1.0 - same_srv_rate)
        host_service_div = float(unique_services / max(1, n_flows))
        traffic_volume_log = float(np.log1p(np.sum(src_bytes) + np.sum(dst_bytes)))

        vals = [
            attack_rate,
            dos_rate,
            probe_rate,
            r2l_rate,
            u2r_rate,
            mean_src_bytes,
            mean_dst_bytes,
            mean_duration,
            mean_count,
            mean_srv_count,
            mean_serror_rate,
            mean_rerror_rate,
            same_srv_rate,
            diff_srv_rate,
            host_service_div,
            traffic_volume_log
        ]
        return np.asarray(vals, dtype=np.float32)


class LivePacketStreamGenerator:
    """Generates synthetic high-fidelity packet streams for live demonstration."""

    def __init__(self, target_ip: str = "192.168.1.100"):
        self.target_ip = target_ip
        self.active_flows: Dict[Tuple, FlowTracker] = {}

    def generate_packet_burst(self, attack_pattern: str = "NORMAL", burst_size: int = 15) -> List[PacketRecord]:
        packets = []
        now = time.time()

        for i in range(burst_size):
            ts = now + (i * 0.05)

            if attack_pattern == "NORMAL":
                src_ip = f"10.0.0.{random.randint(2, 50)}"
                dst_port = random.choice([80, 443, 53])
                pkt = PacketRecord(
                    src_ip=src_ip, dst_ip=self.target_ip,
                    src_port=random.randint(1024, 65535), dst_port=dst_port,
                    protocol='TCP', payload_bytes=random.randint(100, 1500),
                    tcp_flags={'SYN': False, 'FIN': False, 'RST': False, 'ACK': True, 'PSH': True},
                    timestamp=ts
                )

            elif attack_pattern == "PROBE_RECON":
                src_ip = "192.168.1.250"  # Attacker IP
                dst_port = random.randint(1, 1024)  # Port scan
                pkt = PacketRecord(
                    src_ip=src_ip, dst_ip=self.target_ip,
                    src_port=54321, dst_port=dst_port,
                    protocol='TCP', payload_bytes=0,
                    tcp_flags={'SYN': True, 'FIN': False, 'RST': True, 'ACK': False, 'PSH': False},
                    timestamp=ts
                )

            elif attack_pattern == "DOS_FLOOD":
                src_ip = f"172.16.0.{random.randint(1, 254)}"
                pkt = PacketRecord(
                    src_ip=src_ip, dst_ip=self.target_ip,
                    src_port=random.randint(1024, 65535), dst_port=80,
                    protocol='TCP', payload_bytes=40,
                    tcp_flags={'SYN': True, 'FIN': False, 'RST': False, 'ACK': False, 'PSH': False},
                    timestamp=ts
                )

            elif attack_pattern == "R2L_BRUTEFORCE":
                src_ip = "203.0.113.45"
                pkt = PacketRecord(
                    src_ip=src_ip, dst_ip=self.target_ip,
                    src_port=random.randint(1024, 65535), dst_port=22,  # SSH
                    protocol='TCP', payload_bytes=random.randint(300, 800),
                    tcp_flags={'SYN': False, 'FIN': False, 'RST': False, 'ACK': True, 'PSH': True},
                    timestamp=ts
                )

            else:  # U2R_EXFILTRATION
                src_ip = "10.0.0.88"
                pkt = PacketRecord(
                    src_ip=src_ip, dst_ip=self.target_ip,
                    src_port=random.randint(1024, 65535), dst_port=443,
                    protocol='TCP', payload_bytes=random.randint(10000, 60000),
                    tcp_flags={'SYN': False, 'FIN': False, 'RST': False, 'ACK': True, 'PSH': True},
                    timestamp=ts
                )

            packets.append(pkt)

            # Ingest packet into flow tracker
            key = pkt.flow_key
            if key not in self.active_flows:
                self.active_flows[key] = FlowTracker(key)
            self.active_flows[key].add_packet(pkt)

        return packets
