from gridsentinel.simulator.topology import build_fleet


def test_default_fleet_shape():
    f = build_fleet()
    assert len(f.nodes) == 500
    assert f.total_gpus == 4000
    assert len({n.zone_id for n in f.nodes}) == 5
    assert len({n.rack_id for n in f.nodes}) == 100


def test_zone_grouping():
    f = build_fleet()
    z0 = f.by_zone("zone-0")
    assert len(z0) == 100  # 20 racks × 5 nodes
    assert all(n.zone_id == "zone-0" for n in z0)
