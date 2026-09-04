"""Unit tests for OracleShield Multi-Step Autoregressive Rollout & Transformer World Model."""

import unittest
import numpy as np
import torch
from oracle_shield_world_model import (
    WorldModel,
    TransformerWorldModel,
    MultiStepRolloutEngine
)


class TestMultiStepWorldModel(unittest.TestCase):

    def setUp(self):
        self.d_in = 16
        self.seq_len = 8
        self.dummy_seq = np.random.randn(self.seq_len, self.d_in).astype(np.float32)

    def test_lstm_world_model_forward(self):
        model = WorldModel(self.d_in)
        x = torch.randn(2, self.seq_len, self.d_in)
        next_s, stage, z = model(x)

        self.assertEqual(next_s.shape, (2, self.d_in))
        self.assertEqual(stage.shape, (2, 5))
        self.assertEqual(z.shape, (2, 64))

    def test_transformer_world_model_forward(self):
        model = TransformerWorldModel(self.d_in)
        x = torch.randn(2, self.seq_len, self.d_in)
        next_s, stage, z = model(x)

        self.assertEqual(next_s.shape, (2, self.d_in))
        self.assertEqual(stage.shape, (2, 5))
        self.assertEqual(z.shape, (2, 64))

    def test_multi_step_rollout_engine_lstm(self):
        model = WorldModel(self.d_in)
        horizon = 5
        results = MultiStepRolloutEngine.predict_rollout(model, self.dummy_seq, horizon=horizon)

        self.assertEqual(len(results), horizon)
        for i, step_data in enumerate(results):
            self.assertEqual(step_data['step'], i + 1)
            self.assertEqual(step_data['projected_state'].shape, (self.d_in,))
            self.assertIn('predicted_stage', step_data)
            self.assertIn('cumulative_risk', step_data)

    def test_multi_step_rollout_engine_transformer(self):
        model = TransformerWorldModel(self.d_in)
        horizon = 5
        results = MultiStepRolloutEngine.predict_rollout(model, self.dummy_seq, horizon=horizon)

        self.assertEqual(len(results), horizon)
        for i, step_data in enumerate(results):
            self.assertEqual(step_data['step'], i + 1)
            self.assertEqual(step_data['projected_state'].shape, (self.d_in,))
            self.assertIn('predicted_stage', step_data)
            self.assertIn('cumulative_risk', step_data)


if __name__ == "__main__":
    unittest.main()
