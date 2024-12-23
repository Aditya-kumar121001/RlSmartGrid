# feature_extraction.py

import json
from collections import defaultdict, deque
import os

class FeatureExtractor:
    def __init__(self):
        """
        Initializes the FeatureExtractor.
        """
        pass  # No initialization needed since we're not using scalers or complex structures

    def load_physical_data(self, filepath):
        """
        Loads physical network data from a JSON file.

        Args:
            filepath (str): Path to the physical_network.json file.

        Returns:
            Tuple[List[dict], List[dict]]: Lists of physical nodes and physical links.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        with open(filepath, 'r') as f:
            physical_data = json.load(f)
            physical_nodes = physical_data['NP']
            physical_links = physical_data['LP']
        return physical_nodes, physical_links

    def build_adjacency_list(self, physical_links):
        """
        Constructs an adjacency list from physical links.

        Args:
            physical_links (List[dict]): List of physical link dictionaries.

        Returns:
            defaultdict: Adjacency list representation of the network.
        """
        adjacency = defaultdict(list)
        for link in physical_links:
            source = int(link['source'].split('_')[1])
            target = int(link['target'].split('_')[1])
            adjacency[source].append(target)
            adjacency[target].append(source)
        return adjacency

    def get_node_computing_resources(self, physical_nodes):
        """
        Extracts CPU capacities from physical nodes.

        Args:
            physical_nodes (List[dict]): List of physical node dictionaries.

        Returns:
            List[float]: List of CPU capacities.
        """
        return [node['cpu_capacity'] for node in physical_nodes]

    def get_adjacent_link_bandwidth(self, physical_nodes, physical_links):
        """
        Calculates the total adjacent link bandwidth for each node.

        Args:
            physical_nodes (List[dict]): List of physical node dictionaries.
            physical_links (List[dict]): List of physical link dictionaries.

        Returns:
            List[int]: List of total adjacent link bandwidths.
        """
        num_nodes = len(physical_nodes)
        adj_bandwidth = [0 for _ in range(num_nodes)]
        for link in physical_links:
            source = int(link['source'].split('_')[1])
            target = int(link['target'].split('_')[1])
            bandwidth = link['bandwidth']
            adj_bandwidth[source] += bandwidth
            adj_bandwidth[target] += bandwidth
        return adj_bandwidth

    def get_distance_correlation(self, physical_nodes, adjacency, mapped_nodes):
        """
        Computes the average distance from each node to all mapped nodes.

        Args:
            physical_nodes (List[dict]): List of physical node dictionaries.
            adjacency (defaultdict): Adjacency list of the network.
            mapped_nodes (List[int]): List of currently mapped node indices.

        Returns:
            List[float]: List of average distances to mapped nodes.
        """
        num_nodes = len(physical_nodes)
        distance_corr = []
        for k in range(num_nodes):
            # BFS to find shortest paths from node k
            visited = [False] * num_nodes
            distance = [0] * num_nodes
            queue = deque()
            queue.append(k)
            visited[k] = True
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        distance[neighbor] = distance[current] + 1
                        queue.append(neighbor)
            # Extract distances to mapped nodes
            mapped_distances = [distance[mapped_node] for mapped_node in mapped_nodes if mapped_node != k]
            if mapped_distances:
                avg_distance = sum(mapped_distances) / len(mapped_distances)
            else:
                avg_distance = 0.0  # No mapped nodes
            distance_corr.append(avg_distance)
        return distance_corr

    def get_time_correlation(self, physical_nodes, physical_links):
        """
        Computes the average delay from each node to all other nodes.

        Args:
            physical_nodes (List[dict]): List of physical node dictionaries.
            physical_links (List[dict]): List of physical link dictionaries.

        Returns:
            List[float]: List of average delays.
        """
        num_nodes = len(physical_nodes)
        total_delay = [0 for _ in range(num_nodes)]
        link_count = [0 for _ in range(num_nodes)]
        for link in physical_links:
            source = int(link['source'].split('_')[1])
            target = int(link['target'].split('_')[1])
            delay = link['delay']
            total_delay[source] += delay
            total_delay[target] += delay
            link_count[source] += 1
            link_count[target] += 1
        avg_delay = []
        for i in range(num_nodes):
            if link_count[i] > 0:
                avg = total_delay[i] / link_count[i]
            else:
                avg = 0.0
            avg_delay.append(avg)
        return avg_delay

    def get_node_security(self, physical_nodes):
        """
        Extracts security levels from physical nodes.

        Args:
            physical_nodes (List[dict]): List of physical node dictionaries.

        Returns:
            List[int]: List of security levels.
        """
        return [node['security_level'] for node in physical_nodes]

    def get_node_features_matrix(self, physical_nodes, physical_links, mapped_nodes):
        """
        Extracts and constructs the node features matrix.

        Args:
            physical_nodes (List[dict]): List of physical node dictionaries.
            physical_links (List[dict]): List of physical link dictionaries.
            mapped_nodes (List[int]): List of currently mapped node indices.

        Returns:
            List[dict]: List of node feature dictionaries.
        """
        # Extract Features
        cpu_resources = self.get_node_computing_resources(physical_nodes)  # [num_nodes]
        adj_bandwidth = self.get_adjacent_link_bandwidth(physical_nodes, physical_links)  # [num_nodes]
        security = self.get_node_security(physical_nodes)  # [num_nodes]

        adjacency = self.build_adjacency_list(physical_links)
        distance_corr = self.get_distance_correlation(physical_nodes, adjacency, mapped_nodes)  # [num_nodes]
        time_corr = self.get_time_correlation(physical_nodes, physical_links)  # [num_nodes]

        # Combine all features into a list of dictionaries
        node_features = []
        feature_names = ['CPU Resources', 'Adjacent Bandwidth', 'Distance Correlation', 'Time Correlation', 'Security']
        for idx in range(len(physical_nodes)):
            feature = {
                'Node': f'node_{idx}',
                'CPU Resources': cpu_resources[idx],
                'Adjacent Bandwidth': adj_bandwidth[idx],
                'Distance Correlation': distance_corr[idx],
                'Time Correlation': time_corr[idx],
                'Security': security[idx]
            }
            node_features.append(feature)

        return node_features
