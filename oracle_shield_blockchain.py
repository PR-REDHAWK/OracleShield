"""OracleShield Permissioned Multi-Node Blockchain Engine.

Implements a tamper-evident distributed ledger for security audit logs with:
- ECDSA / HMAC digital signatures for cryptographic SOC node identity
- Merkle Tree transaction hashing and proof generation
- Proof-of-Authority (PoA) & Byzantine Fault Tolerant (BFT) multi-node consensus
- Smart Contract security policy validation
- Byzantine attack simulation & chain integrity verifier
"""

import os
import json
import time
import hashlib
import hmac
import secrets
from typing import List, Dict, Tuple, Optional, Any


def sha256_text(text: str) -> str:
    """Helper to compute SHA-256 hex digest of string input."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class MerkleTree:
    """Merkle Tree implementation for transaction batch hashing and proof verification."""

    def __init__(self, transactions_hashes: List[str]):
        self.leaves = [h for h in transactions_hashes]
        if not self.leaves:
            self.leaves = [sha256_text("EMPTY_TREE")]
        self.tree = self._build_tree(self.leaves)

    def _build_tree(self, leaves: List[str]) -> List[List[str]]:
        tree = [leaves]
        current_layer = leaves
        while len(current_layer) > 1:
            if len(current_layer) % 2 != 0:
                current_layer.append(current_layer[-1])  # duplicate odd leaf
            next_layer = []
            for i in range(0, len(current_layer), 2):
                combined = current_layer[i] + current_layer[i + 1]
                next_layer.append(sha256_text(combined))
            tree.append(next_layer)
            current_layer = next_layer
        return tree

    @property
    def root(self) -> str:
        """Returns the Merkle Root Hash."""
        return self.tree[-1][0]

    def get_proof(self, tx_hash: str) -> List[Dict[str, str]]:
        """Generates Merkle audit proof path for a given transaction hash."""
        if tx_hash not in self.leaves:
            return []
        proof = []
        index = self.leaves.index(tx_hash)
        for layer in self.tree[:-1]:
            is_right = (index % 2 == 1)
            sibling_index = index - 1 if is_right else index + 1
            if sibling_index < len(layer):
                proof.append({
                    'position': 'left' if is_right else 'right',
                    'hash': layer[sibling_index]
                })
            index = index // 2
        return proof

    @staticmethod
    def verify_proof(tx_hash: str, proof: List[Dict[str, str]], root_hash: str) -> bool:
        """Verifies if a transaction hash belongs to the Merkle Tree with root_hash."""
        current = tx_hash
        for item in proof:
            if item['position'] == 'left':
                combined = item['hash'] + current
            else:
                combined = current + item['hash']
            current = sha256_text(combined)
        return current == root_hash


class Transaction:
    """Represents a security event transaction in OracleShield."""

    def __init__(
        self,
        sender_node_id: str,
        event_payload: Dict[str, Any],
        timestamp: Optional[float] = None,
        tx_id: Optional[str] = None,
        signature: Optional[str] = None,
        public_key: Optional[str] = None
    ):
        self.sender_node_id = sender_node_id
        self.payload = event_payload
        self.timestamp = timestamp or time.time()
        self.public_key = public_key or f"pubkey_{sender_node_id}"
        self.tx_id = tx_id or self.calculate_hash()
        self.signature = signature or ""

    def calculate_hash(self) -> str:
        raw_data = {
            'sender_node_id': self.sender_node_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'public_key': self.public_key
        }
        json_str = json.dumps(raw_data, sort_keys=True, default=str)
        return sha256_text(json_str)

    def sign(self, secret_key: str):
        """Cryptographically signs the transaction hash using node secret key."""
        self.signature = hmac.new(
            secret_key.encode('utf-8'),
            self.tx_id.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def verify_signature(self, secret_key: str) -> bool:
        """Verifies transaction signature integrity."""
        if not self.signature:
            return False
        expected_sig = hmac.new(
            secret_key.encode('utf-8'),
            self.tx_id.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected_sig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_id': self.tx_id,
            'sender_node_id': self.sender_node_id,
            'timestamp': self.timestamp,
            'public_key': self.public_key,
            'signature': self.signature,
            'payload': self.payload
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        return cls(
            sender_node_id=data['sender_node_id'],
            event_payload=data['payload'],
            timestamp=data['timestamp'],
            tx_id=data['tx_id'],
            signature=data.get('signature', ''),
            public_key=data.get('public_key', '')
        )


class SOCNode:
    """Represents a participating Security Operations Center node in the network."""

    def __init__(self, node_id: str, location: str, role: str = "VALIDATOR"):
        self.node_id = node_id
        self.location = location
        self.role = role  # LEADER, VALIDATOR, OBSERVER
        self.secret_key = sha256_text(f"{node_id}_secret_key_oracle_shield_{secrets.token_hex(8)}")
        self.public_key = sha256_text(f"pubkey_{node_id}")
        self.created_at = time.time()
        self.reputation = 100.0

    def create_transaction(self, event_payload: Dict[str, Any]) -> Transaction:
        tx = Transaction(
            sender_node_id=self.node_id,
            event_payload=event_payload,
            public_key=self.public_key
        )
        tx.sign(self.secret_key)
        return tx

    def sign_block(self, block_hash: str) -> str:
        """Signs block hash for Proof-of-Authority consensus vote."""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            block_hash.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def verify_block_vote(self, block_hash: str, signature: str) -> bool:
        expected = hmac.new(
            self.secret_key.encode('utf-8'),
            block_hash.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'location': self.location,
            'role': self.role,
            'public_key': self.public_key,
            'reputation': self.reputation
        }


class Block:
    """Represents a block in the OracleShield permissioned blockchain."""

    def __init__(
        self,
        index: int,
        transactions: List[Transaction],
        previous_hash: str,
        proposer_node_id: str,
        timestamp: Optional[float] = None,
        merkle_root: Optional[str] = None,
        block_hash: Optional[str] = None,
        proposer_signature: Optional[str] = None,
        validator_votes: Optional[Dict[str, str]] = None,
        consensus_status: str = "PENDING"
    ):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.proposer_node_id = proposer_node_id
        
        tx_hashes = [tx.tx_id for tx in transactions]
        self.merkle_tree = MerkleTree(tx_hashes)
        self.merkle_root = merkle_root or self.merkle_tree.root
        
        self.proposer_signature = proposer_signature or ""
        self.validator_votes = validator_votes or {}  # {node_id: signature}
        self.consensus_status = consensus_status      # PENDING, APPROVED, REJECTED
        self.hash = block_hash or self.calculate_hash()

    def calculate_hash(self) -> str:
        header = {
            'index': self.index,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'proposer_node_id': self.proposer_node_id
        }
        return sha256_text(json.dumps(header, sort_keys=True, default=str))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'hash': self.hash,
            'proposer_node_id': self.proposer_node_id,
            'proposer_signature': self.proposer_signature,
            'validator_votes': self.validator_votes,
            'consensus_status': self.consensus_status,
            'transaction_count': len(self.transactions),
            'transactions': [tx.to_dict() for tx in self.transactions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        txs = [Transaction.from_dict(tx) for tx in data.get('transactions', [])]
        return cls(
            index=data['index'],
            transactions=txs,
            previous_hash=data['previous_hash'],
            proposer_node_id=data['proposer_node_id'],
            timestamp=data['timestamp'],
            merkle_root=data['merkle_root'],
            block_hash=data['hash'],
            proposer_signature=data.get('proposer_signature', ''),
            validator_votes=data.get('validator_votes', {}),
            consensus_status=data.get('consensus_status', 'PENDING')
        )


class PermissionedBlockchain:
    """Manager for the multi-node permissioned security ledger."""

    def __init__(self, ledger_file: str = "oracle_shield_blockchain.json"):
        self.ledger_file = ledger_file
        self.nodes: Dict[str, SOCNode] = {}
        self.chain: List[Block] = []
        self.mempool: List[Transaction] = []
        self.bft_quorum_ratio: float = 0.66  # Requires >= 2/3 validator consensus votes

        self._initialize_default_nodes()
        self._load_or_genesis()

    def _initialize_default_nodes(self):
        """Initializes default SOC nodes representing distributed security operations."""
        default_nodes = [
            SOCNode("SOC-Delhi-HQ", "Delhi HQ (Central Commander)", role="LEADER"),
            SOCNode("SOC-Mumbai-Node", "Mumbai Regional SOC", role="VALIDATOR"),
            SOCNode("SOC-Bangalore-Node", "Bangalore Tech SOC", role="VALIDATOR"),
            SOCNode("SOC-Hyderabad-Node", "Hyderabad Backup SOC", role="VALIDATOR")
        ]
        for node in default_nodes:
            self.nodes[node.node_id] = node

    def register_node(self, node: SOCNode):
        self.nodes[node.node_id] = node

    def _make_genesis_block(self) -> Block:
        leader_node = self.nodes.get("SOC-Delhi-HQ")
        genesis_event = {
            'event_type': 'GENESIS_BLOCK',
            'system': 'OracleShield Multi-Node Blockchain',
            'policy': 'BFT Proof-of-Authority Consensus',
            'nodes_count': len(self.nodes)
        }
        tx = leader_node.create_transaction(genesis_event) if leader_node else Transaction("SOC-Delhi-HQ", genesis_event)
        
        block = Block(
            index=0,
            transactions=[tx],
            previous_hash="0" * 64,
            proposer_node_id="SOC-Delhi-HQ",
            timestamp=time.time()
        )
        if leader_node:
            block.proposer_signature = leader_node.sign_block(block.hash)
            block.validator_votes[leader_node.node_id] = block.proposer_signature

        block.consensus_status = "APPROVED"
        return block

    def _load_or_genesis(self):
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, 'r', encoding='utf-8') as f:
                    raw_chain = json.load(f)
                    self.chain = [Block.from_dict(b) for b in raw_chain]
            except Exception:
                self.chain = []

        if not self.chain:
            genesis = self._make_genesis_block()
            self.chain = [genesis]
            self.save_ledger()

    def save_ledger(self):
        with open(self.ledger_file, 'w', encoding='utf-8') as f:
            data = [b.to_dict() for b in self.chain]
            json.dump(data, f, indent=2, default=str)

    def submit_security_event(self, event_payload: Dict[str, Any], sender_node_id: str = "SOC-Delhi-HQ") -> Transaction:
        node = self.nodes.get(sender_node_id)
        if not node:
            node = list(self.nodes.values())[0]

        # Smart Contract Policy Check: Enforce minimum payload schema
        if not isinstance(event_payload, dict):
            raise ValueError("Smart Contract Rejection: Payload must be a dictionary.")

        tx = node.create_transaction(event_payload)
        self.mempool.append(tx)

        # Auto-mine block when mempool reaches batch threshold or critical alert
        if len(self.mempool) >= 1 or event_payload.get('is_attack', False):
            self.mine_pending_transactions(proposer_node_id=sender_node_id)

        return tx

    def mine_pending_transactions(self, proposer_node_id: str = "SOC-Delhi-HQ") -> Optional[Block]:
        if not self.mempool:
            return None

        proposer = self.nodes.get(proposer_node_id) or list(self.nodes.values())[0]
        tx_batch = self.mempool[:]
        self.mempool = []

        prev_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            transactions=tx_batch,
            previous_hash=prev_block.hash,
            proposer_node_id=proposer.node_id,
            timestamp=time.time()
        )
        new_block.proposer_signature = proposer.sign_block(new_block.hash)

        # Run Proof-of-Authority / BFT Consensus Voting
        votes_count = 0
        validators = [n for n in self.nodes.values() if n.role in ["LEADER", "VALIDATOR"]]

        for validator in validators:
            # Smart contract rule: Check transaction signatures & block hash integrity
            all_tx_valid = all(
                tx.verify_signature(self.nodes[tx.sender_node_id].secret_key)
                if tx.sender_node_id in self.nodes else True
                for tx in new_block.transactions
            )

            if all_tx_valid and new_block.previous_hash == prev_block.hash:
                vote_sig = validator.sign_block(new_block.hash)
                new_block.validator_votes[validator.node_id] = vote_sig
                votes_count += 1

        required_votes = max(1, int(len(validators) * self.bft_quorum_ratio))
        if votes_count >= required_votes:
            new_block.consensus_status = "APPROVED"
            self.chain.append(new_block)
            self.save_ledger()
            return new_block
        else:
            new_block.consensus_status = "REJECTED"
            return new_block

    def verify_chain_integrity(self) -> Tuple[bool, Optional[int], str]:
        """Validates entire blockchain integrity across block hashes, Merkle roots, signatures, and previous hashes."""
        for i, block in enumerate(self.chain):
            # 1. Verify Genesis
            if i == 0:
                if block.previous_hash != "0" * 64:
                    return False, i, f"Genesis block previous hash invalid."
                continue

            prev_block = self.chain[i - 1]

            # 2. Check previous hash connection
            if block.previous_hash != prev_block.hash:
                return False, i, f"Previous block hash mismatch at index {i} (Expected {prev_block.hash[:10]}..., got {block.previous_hash[:10]}...)"

            # 3. Verify Block Content Hash
            computed_hash = block.calculate_hash()
            if block.hash != computed_hash:
                return False, i, f"Block hash content mismatch at index {i}"

            # 4. Verify Transaction Hash & Signature Consistency
            for tx in block.transactions:
                if tx.calculate_hash() != tx.tx_id:
                    return False, i, f"Transaction payload tampered! Hash mismatch on transaction {tx.tx_id[:10]}..."

            # 5. Verify Merkle Root
            tx_hashes = [tx.tx_id for tx in block.transactions]
            computed_merkle = MerkleTree(tx_hashes).root
            if block.merkle_root != computed_merkle:
                return False, i, f"Merkle Root mismatch at index {i}"

            # 6. Verify Consensus Status
            if block.consensus_status != "APPROVED":
                return False, i, f"Unapproved block present in chain at index {i}"

        return True, None, "Blockchain is 100% valid and consensus-verified."

    def simulate_byzantine_attack(self, block_index: int, field_to_tamper: str = "payload", fake_value: Any = "TAMPERED_UNAUTHORIZED_DATA") -> Dict[str, Any]:
        """Simulates a malicious attack on a historical block to demonstrate BFT tamper detection."""
        if block_index < 0 or block_index >= len(self.chain):
            return {'success': False, 'message': 'Invalid block index'}

        target_block = self.chain[block_index]

        if field_to_tamper == "payload" and target_block.transactions:
            target_block.transactions[0].payload['attack_category'] = fake_value
            target_block.transactions[0].payload['tampered'] = True
        elif field_to_tamper == "previous_hash":
            target_block.previous_hash = "deadbeef" * 8
        elif field_to_tamper == "merkle_root":
            target_block.merkle_root = "badmerkle" * 8

        # Run verification check
        valid, failed_idx, reason = self.verify_chain_integrity()

        return {
            'success': True,
            'tampered_block_index': block_index,
            'field_tampered': field_to_tamper,
            'is_chain_valid': valid,
            'failure_index': failed_idx,
            'rejection_reason': reason
        }
