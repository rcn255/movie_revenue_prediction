import sys
import networkx as nx
import os
import json
from networkx.readwrite import json_graph
from two_kg_creation import save_kg

def combine_kg(G1, G2):

    # Merging G2 into G1
    G_combined = G1.copy()

    for node in G2.nodes():
        if node not in G_combined:
            combined_graph.add_node(node)

    # Adding edges from G2 (only if they don't already exist)
    for u, v in G2.edges():
        if not G_combined.has_edge(u, v):
            G_combined.add_edge(u, v)

    return G_combined



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Use python merge_kg.py <kg_name_1> <kg_name_2> <kg_name_combined>")
        sys.exit(1)

    G1_name = sys.argv[1]  # example: 'kg_base' (will be saved in the 'kg' folder)
    G2_name = sys.argv[2]  # example: 'kg_base' (will be saved in the 'kg' folder)
    G_combined_name = sys.argv[3] # example: 'kg_base' (will be saved in the 'kg' folder)

    with open("kg/" + G1_name + ".json") as f:
        data = json.load(f)
        G1 = json_graph.node_link_graph(data)

    with open("kg/" + G2_name + ".json") as f:
        data = json.load(f)
        G2 = json_graph.node_link_graph(data)

    G_combined = combine_kg(G1, G2)
    save_kg(G_combined, G_combined_name)

    print(f"Finished combining {G1_name} and {G2_name} - saved to {G_combined_name}")
