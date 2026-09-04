"""Unit tests for OracleShield MITRE ATT&CK Engine & Automated Playbooks."""

import unittest
import numpy as np
from mitre_engine import MITREMappingEngine, MITRE_DB


class TestMITREEngine(unittest.TestCase):

    def test_mitre_db_integrity(self):
        self.assertIn('T1046', MITRE_DB)
        self.assertIn('T1498', MITRE_DB)
        self.assertIn('T1110', MITRE_DB)
        self.assertIn('T1041', MITRE_DB)

        tech = MITRE_DB['T1498']
        self.assertEqual(tech.technique_id, 'T1498')
        self.assertGreater(len(tech.mitigation_playbook), 0)

    def test_dos_technique_identification(self):
        # High DoS rate state vector
        state = np.zeros(16, dtype=np.float32)
        state[0] = 0.9  # attack_rate
        state[1] = 0.85 # dos_rate
        state[10] = 0.95 # syn_error_rate

        matches = MITREMappingEngine.identify_techniques(state, predicted_stage="Impact / Disruption")
        tech_ids = [m['technique']['technique_id'] for m in matches]

        self.assertIn('T1498', tech_ids)
        self.assertGreater(matches[0]['confidence'], 0.5)

    def test_recon_technique_identification(self):
        # High probe rate state vector
        state = np.zeros(16, dtype=np.float32)
        state[0] = 0.7  # attack_rate
        state[2] = 0.75 # probe_rate
        state[11] = 0.80 # rerror_rate

        matches = MITREMappingEngine.identify_techniques(state, predicted_stage="Reconnaissance")
        tech_ids = [m['technique']['technique_id'] for m in matches]

        self.assertIn('T1046', tech_ids)

    def test_benign_activity(self):
        state = np.zeros(16, dtype=np.float32)
        matches = MITREMappingEngine.identify_techniques(state, predicted_stage="Benign")

        self.assertEqual(matches[0]['technique']['technique_id'], 'BENIGN')


if __name__ == "__main__":
    unittest.main()
