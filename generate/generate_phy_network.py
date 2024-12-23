import random
import numpy as np
import json

def generate_physical_network(num_nodes=100, num_links=550):
    # Generate nodes
    physical_nodes = []
    for i in range(num_nodes):
        node = {
            "id": f"node_{i}",
            "cpu_capacity": random.randint(50, 80),  # U(50,80)
            "security_level": random.randint(1, 3)   # U(1,3)
        }
        physical_nodes.append(node)

    # Generate links
    physical_links = []
    edges = set()
    while len(physical_links) < num_links:
        source = random.randint(0, num_nodes - 1)
        target = random.randint(0, num_nodes - 1)
        if source != target and (source, target) not in edges and (target, source) not in edges:
            link = {
                "source": f"node_{source}",
                "target": f"node_{target}",
                "bandwidth": random.randint(50, 80),  # U(50,80)
                "delay": random.randint(1, 50)        # U(1,50)
            }
            physical_links.append(link)
            edges.add((source, target))

    physical_network = {
        "NP": physical_nodes,
        "LP": physical_links
    }

    # Save to JSON file
    with open('physical_network.json', 'w') as f:
        json.dump(physical_network, f)


# Generate physical network
generate_physical_network()


