"""Fleet topology — racks, zones, power feeds, network.

A node's correlation keys (rack, zone, feed, spine) are what let failures
correlate realistically. Cooling fails per zone; PSU trips per rack; network
flaps per ToR / spine.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    node_id: str
    gpu_idx: int

    @property
    def gpu_id(self) -> str:
        return f"{self.node_id}/gpu{self.gpu_idx}"


@dataclass(frozen=True)
class Node:
    node_id: str
    rack_id: str
    zone_id: str
    feed_id: str
    spine_id: str
    gpus_per_node: int

    @property
    def gpus(self) -> list[GPU]:
        return [GPU(self.node_id, i) for i in range(self.gpus_per_node)]


@dataclass(frozen=True)
class Fleet:
    nodes: list[Node]

    @property
    def total_gpus(self) -> int:
        return sum(n.gpus_per_node for n in self.nodes)

    def by_zone(self, zone_id: str) -> list[Node]:
        return [n for n in self.nodes if n.zone_id == zone_id]

    def by_rack(self, rack_id: str) -> list[Node]:
        return [n for n in self.nodes if n.rack_id == rack_id]

    def by_feed(self, feed_id: str) -> list[Node]:
        return [n for n in self.nodes if n.feed_id == feed_id]


def build_fleet(
    n_nodes: int = 500,
    gpus_per_node: int = 8,
    nodes_per_rack: int = 5,
    racks_per_zone: int = 20,
    feeds_per_zone: int = 2,
    spines: int = 2,
) -> Fleet:
    """Default layout: 500 nodes → 100 racks → 5 zones, 2 power feeds/zone, 2 spines."""
    nodes: list[Node] = []
    for i in range(n_nodes):
        rack_num = i // nodes_per_rack
        zone_num = rack_num // racks_per_zone
        feed_num = (rack_num % racks_per_zone) % feeds_per_zone
        spine_num = rack_num % spines
        nodes.append(
            Node(
                node_id=f"node-{i:04d}",
                rack_id=f"rack-{rack_num:03d}",
                zone_id=f"zone-{zone_num}",
                feed_id=f"zone-{zone_num}/feed-{feed_num}",
                spine_id=f"spine-{spine_num}",
                gpus_per_node=gpus_per_node,
            )
        )
    return Fleet(nodes=nodes)
