"""Unit tests for OracleShield Permissioned Multi-Node Blockchain Engine."""

import os
import unittest
from oracle_shield_blockchain import (
    PermissionedBlockchain,
    SOCNode,
    Block,
    Transaction,
    MerkleTree
)


class TestPermissionedBlockchain(unittest.TestCase):
    
    def setUp(self):
        self.test_ledger = "test_oracle_shield_blockchain.json"
        if os.path.exists(self.test_ledger):
            os.remove(self.test_ledger)
        self.blockchain = PermissionedBlockchain(self.test_ledger)

    def tearDown(self):
        if os.path.exists(self.test_ledger):
            os.remove(self.test_ledger)

    def test_genesis_block(self):
        self.assertEqual(len(self.blockchain.chain), 1)
        genesis = self.blockchain.chain[0]
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.consensus_status, "APPROVED")
        self.assertEqual(genesis.previous_hash, "0" * 64)

    def test_transaction_signature(self):
        node = self.blockchain.nodes["SOC-Delhi-HQ"]
        payload = {"event": "recon_attack", "is_attack": True, "stage": "Reconnaissance"}
        tx = node.create_transaction(payload)
        
        # Verify valid signature
        self.assertTrue(tx.verify_signature(node.secret_key))
        
        # Verify tampered signature fails
        self.assertFalse(tx.verify_signature("wrong_secret_key"))

    def test_merkle_tree_proof(self):
        tx_hashes = ["hash_a", "hash_b", "hash_c", "hash_d"]
        mt = MerkleTree(tx_hashes)
        
        self.assertIsNotNone(mt.root)
        proof = mt.get_proof("hash_b")
        self.assertTrue(MerkleTree.verify_proof("hash_b", proof, mt.root))

    def test_bft_consensus_and_block_mining(self):
        payload = {
            "attack_category": "dos",
            "is_attack": True,
            "stage": "Impact / Disruption",
            "progression_probability": 0.85
        }
        tx = self.blockchain.submit_security_event(payload, sender_node_id="SOC-Delhi-HQ")
        
        self.assertEqual(len(self.blockchain.chain), 2)
        latest_block = self.blockchain.chain[-1]
        self.assertEqual(latest_block.consensus_status, "APPROVED")
        self.assertGreaterEqual(len(latest_block.validator_votes), 3)

        # Verify entire chain integrity
        valid, failed_idx, msg = self.blockchain.verify_chain_integrity()
        self.assertTrue(valid, msg)

    def test_byzantine_attack_simulation(self):
        # Add a block first
        self.blockchain.submit_security_event({"attack": "u2r"}, sender_node_id="SOC-Mumbai-Node")
        
        # Run Byzantine attack simulation
        res = self.blockchain.simulate_byzantine_attack(block_index=1, field_to_tamper="payload", fake_value="TAMPERED")
        
        self.assertTrue(res['success'])
        self.assertFalse(res['is_chain_valid'])
        self.assertEqual(res['failure_index'], 1)


if __name__ == "__main__":
    unittest.main()
