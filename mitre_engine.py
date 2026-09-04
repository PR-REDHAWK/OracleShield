"""OracleShield Granular MITRE ATT&CK Engine & Automated Playbooks.

Maps micro-telemetry, network state vectors, and predicted attack stages directly
to granular MITRE ATT&CK technique IDs, providing actionable automated defense playbooks.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class MITRETechnique:
    """Represents a specific MITRE ATT&CK Technique with metadata and playbook."""

    def __init__(
        self,
        technique_id: str,
        name: str,
        tactic: str,
        description: str,
        mitigation_playbook: List[str],
        recommended_action: str
    ):
        self.technique_id = technique_id
        self.name = name
        self.tactic = tactic
        self.description = description
        self.mitigation_playbook = mitigation_playbook
        self.recommended_action = recommended_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            'technique_id': self.technique_id,
            'name': self.name,
            'tactic': self.tactic,
            'description': self.description,
            'mitigation_playbook': self.mitigation_playbook,
            'recommended_action': self.recommended_action
        }


# Knowledge base of mapped MITRE ATT&CK Techniques
MITRE_DB: Dict[str, MITRETechnique] = {
    'T1046': MITRETechnique(
        technique_id='T1046',
        name='Network Service Discovery',
        tactic='Reconnaissance',
        description='Adversaries attempt to get a listing of services running on remote hosts to identify vulnerable targets.',
        mitigation_playbook=[
            'Apply network segmentation to restrict lateral port scanning.',
            'Deploy dynamic port tarpitting on unused perimeter ports.',
            'Rate-limit ICMP/SYN probes at the edge firewall.'
        ],
        recommended_action='BLOCK_SCANNER_IP'
    ),
    'T1595': MITRETechnique(
        technique_id='T1595',
        name='Active Scanning',
        tactic='Reconnaissance',
        description='Adversaries execute active probing (IP sweeps, SYN scans) to gather network topology intelligence.',
        mitigation_playbook=[
            'Configure IDS/IPS to auto-drop IP sweep bursts.',
            'Obfuscate public-facing service banners.',
            'Enable automated threat intelligence IP feed blocking.'
        ],
        recommended_action='DROP_PERIMETER_PROBES'
    ),
    'T1110': MITRETechnique(
        technique_id='T1110',
        name='Brute Force',
        tactic='Initial Access',
        description='Adversaries systematically attempt multiple password combinations to gain unauthorized access.',
        mitigation_playbook=[
            'Enforce account lockout policies after 5 failed authentication attempts.',
            'Require Multi-Factor Authentication (MFA) for SSH/RDP/FTP services.',
            'Deploy Fail2Ban / IP rate limiting on login endpoints.'
        ],
        recommended_action='ENFORCE_ACCOUNT_LOCKOUT'
    ),
    'T1190': MITRETechnique(
        technique_id='T1190',
        name='Exploit Public-Facing Application',
        tactic='Initial Access',
        description='Adversaries attempt to exploit software vulnerabilities in web servers or public services.',
        mitigation_playbook=[
            'Apply WAF rules to inspect HTTP payloads for exploit signatures.',
            'Isolate web application container in a restricted DMZ network.',
            'Patch web server binaries to latest security release.'
        ],
        recommended_action='TRIGGER_WAF_RULES'
    ),
    'T1068': MITRETechnique(
        technique_id='T1068',
        name='Exploitation for Privilege Escalation',
        tactic='Privilege Escalation',
        description='Adversaries exploit software flaws to elevate operating system access privileges.',
        mitigation_playbook=[
            'Restrict SUID/SGID executable permissions on host endpoints.',
            'Enable Kernel PAM / SELinux strict isolation policy.',
            'Audit privilege elevation system calls via ED/Auditd.'
        ],
        recommended_action='ISOLATE_COMPROMISED_HOST'
    ),
    'T1498': MITRETechnique(
        technique_id='T1498',
        name='Network Denial of Service',
        tactic='Impact / Disruption',
        description='Adversaries flood target network infrastructure with high-volume traffic to exhaust capacity.',
        mitigation_playbook=[
            'Enable TCP SYN Cookies on firewall/load-balancers.',
            'Trigger upstream BGP FlowSpec / Blackholing for flood IP ranges.',
            'Apply rate-limiting on target port HTTP/UDP queues.'
        ],
        recommended_action='ENABLE_SYN_COOKIES_AND_BGP_BLACKHOLE'
    ),
    'T1041': MITRETechnique(
        technique_id='T1041',
        name='Exfiltration Over C2 Channel',
        tactic='Exfiltration',
        description='Adversaries steal sensitive data by transmitting it over established Command and Control channels.',
        mitigation_playbook=[
            'Inspect outbound egress bandwidth anomalies on router interfaces.',
            'Terminate active TLS/SSH C2 sessions matching anomalous volume.',
            'Quarantine target host and initiate incident response triage.'
        ],
        recommended_action='TERMINATE_C2_SESSION'
    )
}


class MITREMappingEngine:
    """Engine mapping 16-D state vectors and flow telemetry to granular MITRE ATT&CK techniques."""

    @classmethod
    def identify_techniques(
        cls,
        state_vector: np.ndarray,
        predicted_stage: str = "Unknown",
        flow_telemetry: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Maps temporal state vectors and predicted stage to MITRE ATT&CK techniques with evidence.
        
        Args:
            state_vector: 16-D numpy array [attack_rate, dos_rate, probe_rate, r2l_rate, u2r_rate, ...]
            predicted_stage: Predicted ATT&CK stage from World Model
            flow_telemetry: Optional dict containing raw flow attributes
        
        Returns:
            List of matched MITRE technique dicts with confidence score and evidence telemetry.
        """
        matches = []
        if state_vector is None or len(state_vector) < 16:
            return matches

        attack_rate = float(state_vector[0])
        dos_rate = float(state_vector[1])
        probe_rate = float(state_vector[2])
        r2l_rate = float(state_vector[3])
        u2r_rate = float(state_vector[4])
        mean_serror = float(state_vector[10])
        mean_rerror = float(state_vector[11])
        host_div = float(state_vector[14])
        traffic_vol = float(state_vector[15])

        # 1. Check Reconnaissance / Active Scanning (T1046 / T1595)
        if probe_rate > 0.15 or mean_rerror > 0.25 or host_div > 0.30:
            tech = MITRE_DB['T1046']
            conf = min(0.99, float(probe_rate * 0.6 + mean_rerror * 0.4 + 0.3))
            matches.append({
                'technique': tech.to_dict(),
                'confidence': conf,
                'evidence': f"Probe Rate={probe_rate:.1%}, RST Error Rate={mean_rerror:.1%}, Service Diversity={host_div:.2f}"
            })

        # 2. Check Network DoS (T1498)
        if dos_rate > 0.20 or mean_serror > 0.30 or (traffic_vol > 8.0 and attack_rate > 0.4):
            tech = MITRE_DB['T1498']
            conf = min(0.99, float(dos_rate * 0.6 + mean_serror * 0.4 + 0.35))
            matches.append({
                'technique': tech.to_dict(),
                'confidence': conf,
                'evidence': f"DoS Rate={dos_rate:.1%}, SYN Error Rate={mean_serror:.1%}, Log Volume={traffic_vol:.2f}"
            })

        # 3. Check Initial Access / Brute Force (T1110 / T1190)
        if r2l_rate > 0.10:
            tech = MITRE_DB['T1110']
            conf = min(0.99, float(r2l_rate * 0.7 + 0.3))
            matches.append({
                'technique': tech.to_dict(),
                'confidence': conf,
                'evidence': f"R2L Attack Rate={r2l_rate:.1%}, Authentication Endpoint Probes Detected"
            })

        # 4. Check Privilege Escalation / Exfiltration (T1068 / T1041)
        if u2r_rate > 0.05 or (traffic_vol > 10.0 and attack_rate > 0.3):
            tech = MITRE_DB['T1041']
            conf = min(0.99, float(u2r_rate * 0.8 + 0.4))
            matches.append({
                'technique': tech.to_dict(),
                'confidence': conf,
                'evidence': f"U2R Rate={u2r_rate:.1%}, High Payload Volume={traffic_vol:.2f}"
            })

        # Default fallback for Benign / Low Activity
        if not matches:
            matches.append({
                'technique': {
                    'technique_id': 'BENIGN',
                    'name': 'Normal Network Activity',
                    'tactic': 'None',
                    'description': 'No malicious ATT&CK technique pattern detected.',
                    'mitigation_playbook': ['Maintain standard security posture and firewall logging.'],
                    'recommended_action': 'MONITOR'
                },
                'confidence': 1.0 - attack_rate,
                'evidence': f"Malicious Pressure={attack_rate:.1%}"
            })

        return matches
