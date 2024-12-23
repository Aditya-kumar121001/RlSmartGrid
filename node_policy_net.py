# node_mapping.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, conv_out_channels=16, conv_kernel_size=1):
        """
        Initializes the Policy Network.

        Args:
            input_size (int): Number of input features per node.
            conv_out_channels (int, optional): Number of output channels for convolution. Defaults to 16.
            conv_kernel_size (int, optional): Kernel size for convolution. Defaults to 1.
        """
        super(PolicyNetwork, self).__init__()
        # Convolution layer: 1D convolution over node features
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_out_channels, kernel_size=conv_kernel_size)
        # Fully connected layer to compute scores
        self.fc = nn.Linear(conv_out_channels, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_features, num_nodes].

        Returns:
            torch.Tensor: Output probabilities of shape [batch_size, num_nodes].
        """
        # Apply convolution
        x = self.conv1(x)  # [batch_size, conv_out_channels, num_nodes]
        x = F.relu(x)       # Apply ReLU activation

        # Apply fully connected layer to each node's conv output
        # Reshape to [batch_size * num_nodes, conv_out_channels]
        batch_size, conv_out_channels, num_nodes = x.shape
        x = x.permute(0, 2, 1).contiguous().view(batch_size * num_nodes, conv_out_channels)
        scores = self.fc(x)  # [batch_size * num_nodes, 1]

        # Reshape back to [batch_size, num_nodes]
        scores = scores.view(batch_size, num_nodes)

        # Apply softmax to get probabilities
        probabilities = F.softmax(scores, dim=1)  # [batch_size, num_nodes]

        return probabilities

def map_virtual_node(node_features, policy_net, virtual_node_cpu_requirement):
    """
    Maps a virtual node to a physical node based on the policy network.

    Args:
        node_features (List[dict]): List of node feature dictionaries.
        policy_net (PolicyNetwork): The trained policy network.
        virtual_node_cpu_requirement (float): CPU requirement of the virtual node.

    Returns:
        Tuple[List[float], int]: A tuple containing the list of probabilities and the index of the selected physical node.
    """
    # Prepare input tensor
    # Extract features and convert to tensor
    features = []
    for feature in node_features:
        features.append([
            feature['CPU Resources'],
            feature['Adjacent Bandwidth'],
            feature['Distance Correlation'],
            feature['Time Correlation'],
            feature['Security']
        ])
    features_tensor = torch.tensor(features, dtype=torch.float)  # [num_nodes, num_features]

    # Add batch dimension and permute to [batch_size, num_features, num_nodes]
    input_tensor = features_tensor.unsqueeze(0).permute(0, 2, 1)  # [1, num_features, num_nodes]

    # Forward pass through the policy network
    with torch.no_grad():
        probabilities_tensor = policy_net(input_tensor)  # [1, num_nodes]

    # Convert probabilities to list without using numpy
    probabilities = probabilities_tensor[0].tolist()  # [num_nodes]

    # Print the probabilities
    print("Probabilities of all physical nodes:")
    for idx, prob in enumerate(probabilities):
        print(f"node_{idx}: {prob:.4f}")

    # Filter out nodes that cannot satisfy the CPU requirement
    eligible_nodes = []
    eligible_probs = []
    for idx, feature in enumerate(node_features):
        if feature['CPU Resources'] >= virtual_node_cpu_requirement:
            eligible_nodes.append(idx)
            eligible_probs.append(probabilities[idx])

    if not eligible_nodes:
        raise ValueError("No eligible physical nodes available for mapping the virtual node.")

    # Normalize the eligible probabilities
    total_prob = sum(eligible_probs)
    normalized_probs = [prob / total_prob for prob in eligible_probs]

    # Select a node based on the probabilities
    selected_node = random.choices(eligible_nodes, weights=normalized_probs, k=1)[0]

    return probabilities, selected_node
