from task072_locomotion_proof import gate


def test_gate_contract():
    base = {"zero_fall_ratio": 1.0, "planar_velocity_error": 0.1, "yaw_error": 0.1, "gravity_xy": 0.1,
            "finite": True, "positive_forward_displacement": True, "checkpoint": True,
            "progression": True, "zero_baseline": True, "untrained_baseline": True, "video": True}
    assert gate(base)[0]
    base["video"] = False
    assert not gate(base)[0]
