"""Unit tests for OracleShield Flow Extractor & Telemetry Engine."""

import unittest
import numpy as np
from flow_extractor import (
    PacketRecord,
    FlowTracker,
    FlowStateEncoder,
    LivePacketStreamGenerator
)


class TestFlowExtractor(unittest.TestCase):

    def test_packet_record(self):
        pkt = PacketRecord(
            src_ip="192.168.1.10", dst_ip="10.0.0.1",
            src_port=12345, dst_port=80,
            protocol="TCP", payload_bytes=500
        )
        self.assertEqual(pkt.src_ip, "192.168.1.10")
        self.assertEqual(pkt.dst_port, 80)
        self.assertIsNotNone(pkt.flow_key)

    def test_flow_tracker(self):
        key = ("192.168.1.10", 12345, "10.0.0.1", 80, "TCP")
        tracker = FlowTracker(key)

        pkt1 = PacketRecord("192.168.1.10", "10.0.0.1", 12345, 80, timestamp=100.0, payload_bytes=100)
        pkt2 = PacketRecord("192.168.1.10", "10.0.0.1", 12345, 80, timestamp=100.5, payload_bytes=200)

        tracker.add_packet(pkt1)
        tracker.add_packet(pkt2)

        self.assertEqual(len(tracker.packets), 2)
        self.assertAlmostEqual(tracker.duration, 0.5)
        mean_iat, _ = tracker.inter_arrival_stats
        self.assertAlmostEqual(mean_iat, 0.5)

    def test_flow_state_encoder(self):
        gen = LivePacketStreamGenerator()
        gen.generate_packet_burst("DOS_FLOOD", 40)
        flows = list(gen.active_flows.values())

        state = FlowStateEncoder.encode_flows_to_state(flows)

        self.assertEqual(state.shape, (16,))
        self.assertGreater(state[0], 0.0)  # attack rate > 0
        self.assertGreater(state[1], 0.0)  # dos rate > 0

    def test_recon_probe_generation(self):
        gen = LivePacketStreamGenerator()
        gen.generate_packet_burst("PROBE_RECON", 30)
        flows = list(gen.active_flows.values())

        state = FlowStateEncoder.encode_flows_to_state(flows)

        self.assertEqual(state.shape, (16,))
        self.assertGreater(state[2], 0.0)  # probe rate > 0


if __name__ == "__main__":
    unittest.main()
