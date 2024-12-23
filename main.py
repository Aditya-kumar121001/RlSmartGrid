# main.py

from featureExtraction.node_feature_ext import FeatureExtractor
from node_policy_net import PolicyNetwork, map_virtual_node
import torch

def main():
    # Initialize FeatureExtractor and load data
    extractor = FeatureExtractor()
    physical_network_file = r'C:\Users\iamad\Documents\GitHub\RlSmartGrid\data\physical_network.json'  # Ensure this file exists in the project directory
    physical_nodes, physical_links = extractor.load_physical_data(physical_network_file)

    # Define the virtual network request (as provided by the user)
    virtual_network_request = {
        "index": 1,
        "arrival_time": 6,
        "departure_time": 11.75358518044047,
        "virtual_nodes": [
            {"node": "Virtual_Node_1", "cpu_req": 7, "safety_req": 2},
            {"node": "Virtual_Node_2", "cpu_req": 16, "safety_req": 2},
            {"node": "Virtual_Node_3", "cpu_req": 19, "safety_req": 3},
            {"node": "Virtual_Node_4", "cpu_req": 9, "safety_req": 2},
            {"node": "Virtual_Node_5", "cpu_req": 15, "safety_req": 1}
        ],
        "virtual_links": [
            {"node1": "Virtual_Node_1", "node2": "Virtual_Node_2", "bandwidth_req": 5, "delay_req": 13},
            {"node1": "Virtual_Node_1", "node2": "Virtual_Node_3", "bandwidth_req": 10, "delay_req": 3},
            {"node1": "Virtual_Node_2", "node2": "Virtual_Node_3", "bandwidth_req": 4, "delay_req": 13},
            {"node1": "Virtual_Node_2", "node2": "Virtual_Node_5", "bandwidth_req": 2, "delay_req": 12},
            {"node1": "Virtual_Node_3", "node2": "Virtual_Node_4", "bandwidth_req": 22, "delay_req": 13},
            {"node1": "Virtual_Node_4", "node2": "Virtual_Node_5", "bandwidth_req": 4, "delay_req": 4}
        ]
    }

    # Initialize mapping results
    mapping_results = []

    # Initialize Policy Network
    input_size = 5  # ['CPU Resources', 'Adjacent Bandwidth', 'Distance Correlation', 'Time Correlation', 'Security']
    policy_net = PolicyNetwork(input_size=input_size)
    
    # For demonstration, we'll randomly initialize the network weights
    # In practice, you should train the policy network using reinforcement learning
    policy_net.eval()  # Set to evaluation mode

    # Extract node features
    mapped_nodes = []  # Initially, no nodes are mapped
    node_features = extractor.get_node_features_matrix(
        physical_nodes, physical_links, mapped_nodes
    )

    # Remove or comment out the following block to stop printing physical network features
    # print("\nRaw Node Features:")
    # for feature in node_features:
    #     print(feature)

    # Process each virtual node in the virtual network request
    for v_node in virtual_network_request['virtual_nodes']:
        v_node_name = v_node['node']
        v_cpu_req = v_node['cpu_req']
        v_safety_req = v_node['safety_req']  # Currently unused in mapping

        print(f"\nMapping {v_node_name} with CPU requirement {v_cpu_req} and Safety requirement {v_safety_req}.")

        try:
            # Perform mapping
            probabilities, selected_node_idx = map_virtual_node(node_features, policy_net, v_cpu_req)
            selected_node = physical_nodes[selected_node_idx]

            # Record the mapping
            mapping_results.append({
                "virtual_node": v_node_name,
                "mapped_physical_node": selected_node['id']
            })

            print(f"Selected Physical Node: {selected_node['id']}")

            # Update the physical node's CPU capacity
            physical_nodes[selected_node_idx]['cpu_capacity'] -= v_cpu_req
            if physical_nodes[selected_node_idx]['cpu_capacity'] < 0:
                physical_nodes[selected_node_idx]['cpu_capacity'] = 0  # Prevent negative capacity

            # Update node_features to reflect the reduced CPU capacity
            node_features[selected_node_idx]['CPU Resources'] = physical_nodes[selected_node_idx]['cpu_capacity']

            # Add the node to mapped_nodes if needed for distance correlation
            mapped_nodes.append(selected_node_idx)

        except ValueError as e:
            print(f"Failed to map {v_node_name}: {e}")

    # Display the final mapping results
    print("\nFinal Mapping Results:")
    for result in mapping_results:
        print(f"{result['virtual_node']} --> {result['mapped_physical_node']}")

    # Display unmapped virtual nodes
    mapped_virtual_nodes = [result['virtual_node'] for result in mapping_results]
    all_virtual_nodes = [v_node['node'] for v_node in virtual_network_request['virtual_nodes']]
    unmapped_virtual_nodes = set(all_virtual_nodes) - set(mapped_virtual_nodes)
    if unmapped_virtual_nodes:
        print("\nUnmapped Virtual Nodes:")
        for v_node in unmapped_virtual_nodes:
            print(v_node)
    else:
        print("\nAll virtual nodes have been successfully mapped.")

if __name__ == "__main__":
    main()
