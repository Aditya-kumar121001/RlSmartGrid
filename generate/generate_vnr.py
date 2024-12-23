import random
import numpy as np
import json

def generate_vnrs(num_vnrs=1000):
    vnrs = []
    for idx in range(num_vnrs):
        num_nodes = random.randint(2, 10)  # U(2,10)
        virtual_nodes = []
        for i in range(num_nodes):
            node = {
                "node": f"Virtual_Node_{i+1}",
                "cpu_req": random.randint(1, 30),       # U(1,30)
                "safety_req": random.randint(1, 3)      # U(1,3)
            }
            virtual_nodes.append(node)


        virtual_links = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if random.random() < 0.5:  # 50% connection probability
                    link = {
                        "node1": f"Virtual_Node_{i+1}",
                        "node2": f"Virtual_Node_{j+1}",
                        "bandwidth_req": random.randint(1, 30),  # U(1,30)
                        "delay_req": random.randint(1, 20)       # U(1,20)
                    }
                    virtual_links.append(link)

        arrival_time = np.random.poisson(lam=5)
        duration = np.random.exponential(scale=10)
        departure_time = arrival_time + duration

        vnr = {
            "index": idx + 1,
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "virtual_nodes": virtual_nodes,
            "virtual_links": virtual_links
        }
        vnrs.append(vnr)
    return vnrs

# Generate training and testing VNRs
training_vnrs = generate_vnrs(1000) 
testing_vnrs = generate_vnrs(1000)   

# Save VNRs to JSON files
with open('training_vnrs.json', 'w') as f:
    json.dump(training_vnrs, f)

with open('testing_vnrs.json', 'w') as f:
    json.dump(testing_vnrs, f)