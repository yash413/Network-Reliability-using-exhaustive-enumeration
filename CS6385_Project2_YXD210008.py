# Import the required libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Function for calulating the final reliability value of the network
def network_reliability(p, num_nodes):
    reliability_val = 0
    for state in all_states(num_nodes):
        if if_operational(state, num_nodes):
            reliability_val += state_probability(state, p)
    reliability = reliability_val
    return reliability

# Function for returning all the triangle states possible
def all_states(num_nodes):
    # Generate all possible combinations of up/down states for triangles
    # Each triangle is represented as 1 for up and 0 for down
    num_triangles = num_nodes * (num_nodes - 1) * (num_nodes - 2) // 6
    triangle_states = list(itertools.product([0, 1], repeat=num_triangles))
    
    return triangle_states

# Function to check if the given state is operational
def if_operational(state, num_nodes):
    # Check if the network is operational for a given state
    # Return True if operational, False otherwise
    adjacency_matrix = state_to_adjacency_matrix(state, num_nodes)
    
    # Check if there are any isolated nodes
    return not any(np.sum(adjacency_matrix, axis=0) == 0)

# Function to convert the given state to an adjacency matrix
def state_to_adjacency_matrix(state, num_nodes):
    # Initializing the adjacency matrix
    adjacency_matrix = np.ones((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        adjacency_matrix[i][i] = 0

    edges = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    
    # Making the edges of the failing triangle 0 in the matrix
    for l in range(len(state)):
      (i, j, k) = edges[l]
      if state[l] == 0:
        adjacency_matrix[i][j] = adjacency_matrix[j][i] = 0
        adjacency_matrix[j][k] = adjacency_matrix[k][j] = 0
        adjacency_matrix[i][k] = adjacency_matrix[k][i] = 0

    return adjacency_matrix

# Function to calculate the probability of the given state
def state_probability(state, p):
    # Calculate the probability of a given state based on the failure probability p
    probability = 1.0
    
    for triangle_state in state:
        probability *= p if triangle_state == 1 else (1 - p)
    
    return probability


# Main Program
num_nodes = 5
probabilities = []
net_reliabilities = []

# Run the program for different values of p and collect reliability values
cur_p = 0.05
while cur_p < 1.05:
  probabilities.append(round(cur_p, 2))
  net_reliabilities.append(network_reliability(cur_p, num_nodes))
  cur_p += 0.05

print("Probability \t Network Reliability")
# Print the reliability values for all the p values
for i in range(len(probabilities)):
  print(f"p = {probabilities[i]} \t {net_reliabilities[i]}")

# Plot the results (Probability vs Network Reliability)
plt.plot(probabilities, net_reliabilities, marker='o')
plt.xlabel('Probabilities (P Values)')
plt.ylabel('Network Reliability')
plt.title('Probabilities vs Network Reliability')
plt.show()