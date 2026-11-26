"""
Smart City Route Optimization System
------------------------------------
Single-file Python project implementing:
1. Weighted city graph representation
2. Dijkstra's shortest-path algorithm (non-negative weights)
3. Bellman-Ford shortest-path algorithm (supports negative weights)
4. Negative-cycle detection
5. Route reconstruction
6. Execution-time comparison on generated graph sizes

Run:
    python smart_city_route_optimization.py

No third-party packages are required.
"""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Node = str
Edge = Tuple[Node, Node, float]


@dataclass
class RouteResult:
    path: Optional[List[Node]]
    distance: float
    execution_time: float


class CityGraph:
    """Weighted graph. Roads are undirected by default."""

    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self.adjacency: Dict[Node, List[Tuple[Node, float]]] = {}

    def add_location(self, location: Node) -> None:
        self.adjacency.setdefault(location, [])

    def add_road(self, start: Node, end: Node, distance: float) -> None:
        if distance < 0 and not self.directed:
            # Negative undirected edges create a negative cycle by traversing
            # the same edge in both directions, so reject them.
            raise ValueError(
                "Negative road weights are not supported for undirected roads. "
                "Use a directed graph to demonstrate Bellman-Ford negative edges."
            )
        self.add_location(start)
        self.add_location(end)
        self.adjacency[start].append((end, distance))
        if not self.directed:
            self.adjacency[end].append((start, distance))

    def get_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for start, neighbors in self.adjacency.items():
            for end, weight in neighbors:
                edges.append((start, end, weight))
        return edges

    def locations(self) -> List[Node]:
        return list(self.adjacency.keys())


def reconstruct_path(
    previous: Dict[Node, Optional[Node]],
    source: Node,
    destination: Node,
) -> Optional[List[Node]]:
    """Reconstruct a route from predecessor information."""
    if source == destination:
        return [source]

    if previous.get(destination) is None:
        return None

    path: List[Node] = []
    current: Optional[Node] = destination

    while current is not None:
        path.append(current)
        if current == source:
            path.reverse()
            return path
        current = previous.get(current)

    return None


def dijkstra(
    graph: CityGraph, source: Node, destination: Node
) -> Tuple[Optional[List[Node]], float]:
    """
    Find a shortest route using a min-priority queue.
    Requires all edge weights to be non-negative.
    Time: O((V + E) log V) with an adjacency list and heap.
    """
    if source not in graph.adjacency or destination not in graph.adjacency:
        raise ValueError("Source or destination is not in the city graph.")

    for _, _, weight in graph.get_edges():
        if weight < 0:
            raise ValueError("Dijkstra cannot process negative edge weights.")

    distances = {node: float("inf") for node in graph.locations()}
    previous: Dict[Node, Optional[Node]] = {node: None for node in graph.locations()}
    distances[source] = 0.0

    priority_queue: List[Tuple[float, Node]] = [(0.0, source)]

    while priority_queue:
        current_distance, current = heapq.heappop(priority_queue)

        # Ignore an outdated queue entry.
        if current_distance > distances[current]:
            continue

        if current == destination:
            break

        for neighbor, weight in graph.adjacency[current]:
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current
                heapq.heappush(priority_queue, (new_distance, neighbor))

    if distances[destination] == float("inf"):
        return None, float("inf")

    return reconstruct_path(previous, source, destination), distances[destination]


def bellman_ford(
    graph: CityGraph, source: Node, destination: Node
) -> Tuple[Optional[List[Node]], float]:
    """
    Find a shortest route by relaxing every edge V-1 times.
    Supports negative edge weights and detects reachable negative cycles.
    Time: O(V * E).
    """
    if source not in graph.adjacency or destination not in graph.adjacency:
        raise ValueError("Source or destination is not in the city graph.")

    nodes = graph.locations()
    edges = graph.get_edges()

    distances = {node: float("inf") for node in nodes}
    previous: Dict[Node, Optional[Node]] = {node: None for node in nodes}
    distances[source] = 0.0

    for _ in range(len(nodes) - 1):
        changed = False

        for start, end, weight in edges:
            if distances[start] != float("inf"):
                candidate = distances[start] + weight
                if candidate < distances[end]:
                    distances[end] = candidate
                    previous[end] = start
                    changed = True

        # Stop early if no distance was updated in this pass.
        if not changed:
            break

    # A further possible improvement means a reachable negative cycle exists.
    for start, end, weight in edges:
        if (
            distances[start] != float("inf")
            and distances[start] + weight < distances[end]
        ):
            raise ValueError("A reachable negative-weight cycle was detected.")

    if distances[destination] == float("inf"):
        return None, float("inf")

    return reconstruct_path(previous, source, destination), distances[destination]


def timed_route(
    algorithm, graph: CityGraph, source: Node, destination: Node
) -> RouteResult:
    start_time = time.perf_counter()
    path, distance = algorithm(graph, source, destination)
    elapsed = time.perf_counter() - start_time
    return RouteResult(path, distance, elapsed)


def build_sample_city() -> CityGraph:
    """Create a small example city with distances measured in kilometres."""
    city = CityGraph(directed=False)

    roads = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
    ]

    for start, end, distance in roads:
        city.add_road(start, end, distance)

    return city


def display_route(label: str, result: RouteResult) -> None:
    print(f"\n{label}")
    if result.path is None:
        print("Route: No route exists.")
        print("Distance: Unreachable")
    else:
        print("Route:", " -> ".join(result.path))
        print(f"Distance: {result.distance:g} km")
    print(f"Execution time: {result.execution_time:.8f} seconds")


def run_route_finder() -> None:
    """Ask the user for locations and compare both algorithms."""
    city = build_sample_city()

    print("\n========== SMART CITY ROUTE OPTIMIZATION ==========")
    print("Available locations:", ", ".join(city.locations()))
    print("Road distances are measured in kilometres.")

    source = input("Enter starting location: ").strip().upper()
    destination = input("Enter destination: ").strip().upper()

    if source not in city.adjacency or destination not in city.adjacency:
        print("Invalid location. Please choose from the available locations.")
        return

    dijkstra_result = timed_route(dijkstra, city, source, destination)
    bellman_result = timed_route(bellman_ford, city, source, destination)

    display_route("----- DIJKSTRA ALGORITHM -----", dijkstra_result)
    display_route("----- BELLMAN-FORD ALGORITHM -----", bellman_result)

    if dijkstra_result.distance == bellman_result.distance:
        print("\nResult: Both algorithms found the same shortest distance.")
    else:
        print("\nWarning: The algorithms returned different distances.")


def generate_connected_graph(
    vertex_count: int, extra_edges_per_vertex: int = 3, seed: int = 42
) -> CityGraph:
    """
    Generate a connected, undirected graph with positive weights.
    A chain guarantees connectivity; random extra roads make it denser.
    """
    rng = random.Random(seed + vertex_count)
    graph = CityGraph(directed=False)
    nodes = [f"L{i}" for i in range(vertex_count)]

    for node in nodes:
        graph.add_location(node)

    # Add a chain so every location is reachable.
    for i in range(vertex_count - 1):
        graph.add_road(nodes[i], nodes[i + 1], rng.randint(1, 30))

    # Add random roads.
    possible_edges = set()
    target_extra_edges = min(
        vertex_count * extra_edges_per_vertex,
        vertex_count * (vertex_count - 1) // 2 - (vertex_count - 1),
    )

    while len(possible_edges) < target_extra_edges:
        a = rng.randrange(vertex_count)
        b = rng.randrange(vertex_count)
        if a == b:
            continue
        edge = (min(a, b), max(a, b))
        if edge not in possible_edges and abs(a - b) != 1:
            possible_edges.add(edge)

    for a, b in possible_edges:
        graph.add_road(nodes[a], nodes[b], rng.randint(1, 30))

    return graph


def benchmark_algorithms() -> None:
    """
    Compare both algorithms on progressively larger generated graphs.
    Timings are machine-dependent; run more than once for a better estimate.
    """
    print("\n========== PERFORMANCE COMPARISON ==========")
    print(
        f"{'Vertices':>10} {'Edges':>10} "
        f"{'Dijkstra (s)':>16} {'Bellman-Ford (s)':>20}"
    )
    print("-" * 62)

    sizes = [10, 50, 100, 250, 500]

    for size in sizes:
        graph = generate_connected_graph(size)
        source = "L0"
        destination = f"L{size - 1}"

        dijkstra_result = timed_route(dijkstra, graph, source, destination)
        bellman_result = timed_route(bellman_ford, graph, source, destination)

        edge_count = len(graph.get_edges()) // 2  # Undirected roads counted once.

        print(
            f"{size:>10} {edge_count:>10} "
            f"{dijkstra_result.execution_time:>16.8f} "
            f"{bellman_result.execution_time:>20.8f}"
        )

        if dijkstra_result.distance != bellman_result.distance:
            print("WARNING: shortest distances do not match for this graph.")


def run_algorithm_demo() -> None:
    """Small directed example showing Bellman-Ford's negative-edge support."""
    graph = CityGraph(directed=True)
    graph.add_road("A", "B", 4)
    graph.add_road("A", "C", 5)
    graph.add_road("B", "C", -2)
    graph.add_road("C", "D", 3)
    graph.add_road("B", "D", 6)

    print("\n========== BELLMAN-FORD NEGATIVE-EDGE DEMO ==========")
    path, distance = bellman_ford(graph, "A", "D")
    print("Directed graph: A -> B (4), A -> C (5), B -> C (-2),")
    print("                C -> D (3), B -> D (6)")
    print("Shortest route:", " -> ".join(path) if path else "No route")
    print(f"Shortest distance: {distance:g}")


def main() -> None:
    while True:
        print("\n========== MAIN MENU ==========")
        print("1. Find shortest route in sample city")
        print("2. Compare algorithm performance")
        print("3. Bellman-Ford negative-edge demonstration")
        print("4. Exit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            run_route_finder()
        elif choice == "2":
            benchmark_algorithms()
        elif choice == "3":
            run_algorithm_demo()
        elif choice == "4":
            print("Thank you for using Smart City Route Optimization System.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
