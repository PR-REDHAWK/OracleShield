import os, json, hashlib, time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    class WorldModel(nn.Module):
        def __init__(self, d_in, hidden=64, classes=5):
            super().__init__()

            self.lstm = nn.LSTM(
                d_in,
                hidden,
                num_layers=1,
                batch_first=True
            )

            self.next_state = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, d_in)
            )

            self.stage = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, classes)
            )

        def forward(self, x):
            h, _ = self.lstm(x)
            z = h[:, -1]
            return self.next_state(z), self.stage(z), z

except Exception:
    WorldModel = None

STAGE_MAP = {
    'normal': 'Benign / No active stage',
    'probe': 'Reconnaissance',
    'r2l': 'Initial Access',
    'u2r': 'Privilege Escalation',
    'dos': 'Impact / Disruption',
}

STATE_NAMES = [
    'attack_rate', 'dos_rate', 'probe_rate', 'r2l_rate', 'u2r_rate',
    'mean_src_bytes', 'mean_dst_bytes', 'mean_duration', 'mean_count',
    'mean_srv_count', 'mean_serror_rate', 'mean_rerror_rate',
    'mean_same_srv_rate', 'mean_diff_srv_rate', 'host_service_diversity',
    'traffic_volume_log'
]

NUMERIC_STATE = [
    'src_bytes','dst_bytes','duration','count','srv_count','serror_rate',
    'rerror_rate','same_srv_rate','diff_srv_rate','dst_host_count',
    'dst_host_srv_count'
]


def _safe_mean(s):
    return float(pd.to_numeric(s, errors='coerce').fillna(0).mean()) if len(s) else 0.0


def state_from_window(df: pd.DataFrame) -> np.ndarray:
    n = max(len(df), 1)

    vc = (
        df['attack_category']
        .value_counts(normalize=True)
        if 'attack_category' in df
        else pd.Series(dtype=float)
    )

    attack_rate = (
        float(df['is_attack'].mean())
        if 'is_attack' in df
        else 0.0
    )

    host_service_div = (
        float(df['service'].nunique() / n)
        if 'service' in df
        else 0.0
    )

    volume = float(
        np.log1p(
            pd.to_numeric(
                df.get(
                    'src_bytes',
                    pd.Series(dtype=float)
                ),
                errors='coerce'
            ).fillna(0).sum()
            +
            pd.to_numeric(
                df.get(
                    'dst_bytes',
                    pd.Series(dtype=float)
                ),
                errors='coerce'
            ).fillna(0).sum()
        )
    )

    # Use normalized attack-category frequencies.
    dos_rate = float(vc.get('dos', 0.0))
    probe_rate = float(vc.get('probe', 0.0))
    r2l_rate = float(vc.get('r2l', 0.0))
    u2r_rate = float(vc.get('u2r', 0.0))

    vals = [

        attack_rate,

        dos_rate,
        probe_rate,
        r2l_rate,
        u2r_rate,

        _safe_mean(
            df.get(
                'src_bytes',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'dst_bytes',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'duration',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'count',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'srv_count',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'serror_rate',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'rerror_rate',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'same_srv_rate',
                pd.Series(dtype=float)
            )
        ),

        _safe_mean(
            df.get(
                'diff_srv_rate',
                pd.Series(dtype=float)
            )
        ),

        host_service_div,

        volume,
    ]

    return np.asarray(
        vals,
        dtype=np.float32
    )


def build_state_series(
    df: pd.DataFrame,
    window_size: int = 50
) -> Tuple[np.ndarray, List[str]]:

    states = []
    labels = []

    for start in range(0, len(df) - window_size + 1):

        w = df.iloc[start:start + window_size]

        states.append(
            state_from_window(w)
        )

        # Label the window using the most security-relevant
        # attack present, rather than majority voting.
        counts = w['attack_category'].value_counts(normalize=True)

        if len(counts) == 0:
            label = 'normal'
        else:
            # Keep the dominant class when it is reasonably strong.
            dominant = str(counts.index[0]).lower()
            dominant_fraction = float(counts.iloc[0])

            # Otherwise preserve minority attacks if they occupy
            # a meaningful fraction of the window.
            if dominant_fraction >= 0.50:
                label = dominant
            else:
                attack_priority = ['u2r', 'r2l', 'probe', 'dos']

                label = 'normal'

                for attack in attack_priority:
                    if float(counts.get(attack, 0.0)) >= 0.02:
                        label = attack
                        break

        labels.append(label)

    return np.stack(states), labels

def stage_for_label(label: str) -> str:
    return STAGE_MAP.get(label, 'Unknown')




def stage_score(label_probs: Dict[str, float]) -> Tuple[str, float]:
    if not label_probs:
        return 'Unknown', 0.0
    k = max(label_probs, key=label_probs.get)
    return stage_for_label(k), float(label_probs[k])


def hash_event(event: Dict, previous_hash: str) -> str:
    payload = json.dumps({'event': event, 'previous_hash': previous_hash}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class PersistentThreatMemory:
    """Persistent, self-supervised threat memory.

    It learns recurring network-state prototypes and transitions from observations.
    It does not use the detector's predictions as ground-truth training labels.
    """
    def __init__(self, path='oracle_shield_memory.json'):
        self.path=path
        self.prototypes=[]
        self.transitions={}
        self.baseline=None
        self.ema=None
        self.count=0
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                d=json.load(open(self.path,'r',encoding='utf-8'))
                self.prototypes=d.get('prototypes',[])
                self.transitions=d.get('transitions',{})
                self.baseline=np.asarray(d['baseline'],dtype=float) if d.get('baseline') is not None else None
                self.ema=np.asarray(d['ema'],dtype=float) if d.get('ema') is not None else None
                self.count=int(d.get('count',0))
            except Exception:
                pass

    def _save(self):
        payload={'prototypes':self.prototypes[-300:], 'transitions':self.transitions,
                 'baseline':None if self.baseline is None else self.baseline.tolist(),
                 'ema':None if self.ema is None else self.ema.tolist(), 'count':self.count}
        json.dump(payload,open(self.path,'w',encoding='utf-8'),indent=2)

    def update(self,state,stage,previous_stage=None):
        x=np.asarray(state,dtype=float)
        self.count+=1
        if self.baseline is None:
            self.baseline=x.copy(); self.ema=x.copy()
        else:
            self.ema=.9*self.ema+.1*x
        novelty=float(np.linalg.norm(x-self.ema)/(np.linalg.norm(self.ema)+1e-6))
        # Store novel states as episodic memory. This is unsupervised: no label is required.
        if novelty>0.08 or len(self.prototypes)<10:
            self.prototypes.append({'state':x.tolist(),'stage':stage,'novelty':novelty,'seen':datetime.now().isoformat(timespec='seconds')})
        if previous_stage:
            key=f'{previous_stage}->{stage}'
            self.transitions[key]=int(self.transitions.get(key,0))+1
        self._save()
        return novelty

    def similarity_to_memory(self,state):
        if not self.prototypes: return 0.0,None
        x=np.asarray(state,dtype=float)
        best=0.0; label=None
        for p in self.prototypes:
            y=np.asarray(p['state'],dtype=float)
            sim=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-8))
            if sim>best: best=sim; label=p.get('stage')
        return best,label


class AuditChain:
    def __init__(self, path='oracle_shield_ledger.json'):
        self.path = path
        self.blocks = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.blocks = json.load(f)
            except Exception:
                self.blocks = []
        if not self.blocks:
            self.blocks = [self._make_genesis()]
            self._save()

    def _make_genesis(self):
        event = {'type': 'genesis', 'system': 'OracleShield', 'created': time.time()}
        return {'index': 0, 'timestamp': time.time(), 'event': event,
                'previous_hash': '0'*64, 'hash': hash_event(event, '0'*64)}

    def _save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.blocks, f, indent=2, default=str)

    def append(self, event: Dict):
        prev = self.blocks[-1]
        block = {
            'index': len(self.blocks), 'timestamp': time.time(), 'event': event,
            'previous_hash': prev['hash'], 'hash': hash_event(event, prev['hash'])
        }
        self.blocks.append(block)
        self._save()
        return block

    def verify(self, blocks=None):
        blocks = blocks or self.blocks
        for i, b in enumerate(blocks):
            prev = '0'*64 if i == 0 else blocks[i-1]['hash']
            if b.get('previous_hash') != prev:
                return False, i, 'previous_hash mismatch'
            expected = hash_event(b.get('event', {}), prev)
            if b.get('hash') != expected:
                return False, i, 'content hash mismatch'
        return True, None, 'chain valid'
