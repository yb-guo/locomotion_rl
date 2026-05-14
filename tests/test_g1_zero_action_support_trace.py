import pytest

from h200_locomotion_lab.tools import g1_zero_action_support_trace as trace


def test_parse_args_defaults_to_task023_hybrid_trace() -> None:
    args = trace.parse_args([])

    assert args.asset_variant == "task023_hybrid"
    assert args.start_step == 80
    assert args.end_step == 130
    assert args.n_envs == 1


def test_support_points_filter_disabled_geoms_and_expand_boxes() -> None:
    xml = """
<mujoco>
  <worldbody>
    <body name="left_ankle_roll_link">
      <geom name="visual" type="mesh" />
      <geom name="disabled" type="sphere" contype="0" conaffinity="0" pos="9 9 9" size="1" />
      <geom name="sphere" type="sphere" pos="0.1 0.2 -0.3" size="0.01" />
      <geom name="box" type="box" pos="1 2 3" size="0.5 0.25 0.1" />
    </body>
  </worldbody>
</mujoco>
"""
    root = trace.ElementTree.fromstring(xml)

    points = trace.support_points_for_body(root, "left_ankle_roll_link")

    assert (0.1, 0.2, -0.3) in points
    assert (0.5, 1.75, 3.0) in points
    assert (1.5, 2.25, 3.0) in points
    assert all(point[0] != 9.0 for point in points)


def test_convex_hull_area_and_signed_margin() -> None:
    hull = trace.convex_hull([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    assert trace.polygon_area(hull) == pytest.approx(1.0)
    assert trace.signed_point_polygon_margin((0.5, 0.5), hull) == pytest.approx(0.5)
    assert trace.signed_point_polygon_margin((1.5, 0.5), hull) == pytest.approx(-0.5)


def test_rotate_vector_by_identity_quat() -> None:
    assert trace.rotate_vector_by_quat((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0)) == pytest.approx(
        (1.0, 2.0, 3.0)
    )


def test_mass_weighted_com_uses_resolved_link_positions() -> None:
    robot = FakeRobot(
        {
            0: (0.0, 0.0, 0.0),
            1: (2.0, 0.0, 0.0),
        }
    )

    com = trace.estimate_com(
        robot=robot,
        mass_model={"a": 1.0, "b": 3.0, "missing": 10.0},
        link_indices={"a": 0, "b": 1, "missing": None},
    )

    assert com == pytest.approx((1.5, 0.0, 0.0))


class FakeRobot:
    def __init__(self, positions: dict[int, tuple[float, float, float]]) -> None:
        self.positions = positions

    def get_links_pos(self, links_idx_local: tuple[int, ...]) -> list[list[float]]:
        return [list(self.positions[index]) for index in links_idx_local]
