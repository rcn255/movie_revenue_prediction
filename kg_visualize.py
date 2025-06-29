import os
import sys
import json
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random

def random_embeddings(graph, dim=64):
    embeddings = {}
    for node in graph.nodes():
        embeddings[node] = np.random.uniform(-1, 1, dim)
    return embeddings

def generate_node_visualization(graph, node_list, dim=64):
    os.makedirs('images', exist_ok=True)

    # Check if any nodes in node_list are movies that don't exist in the graph (for query visualization)
    movie_nodes_to_create = []
    for node in node_list:
        if node not in graph.nodes():
            movie_nodes_to_create.append(node)

    for movie_node in movie_nodes_to_create:
        # Add the movie node to the graph
        graph.add_node(movie_node, type="movie", id=movie_node)

        # Connect to other nodes in node_list (excluding other movies)
        for other_node in node_list:
            if other_node != movie_node and other_node in graph.nodes():
                node_type = graph.nodes[other_node].get("type")

                if node_type == "actor":
                    # Add edge from actor to movie with "acted" role
                    graph.add_edge(other_node, movie_node, role="acted")
                elif node_type == "director":
                    # Add edge from director to movie with "directed" role
                    graph.add_edge(other_node, movie_node, role="directed")

    # Computing subgraph from provided node IDs
    nodes_to_include = set(node_list)
    for node in node_list:
        if node in graph.nodes():  # Check if node exists before getting neighbors
            nodes_to_include.update(graph.neighbors(node))
    subgraph = graph.subgraph(nodes_to_include)

    # Computing embeddings via random embedding
    embeddings = random_embeddings(subgraph, dim=dim)

    # Reduce to 2D using TSNE
    # Start: Generated via ChatGPT (Prompt: How to reduce transe to 2 dim and apply to networkx graph)
    all_nodes = list(subgraph.nodes())
    X = np.array([embeddings[node] for node in all_nodes])
    if len(all_nodes) == 1:
        # Single node fallback
        pos_2d = {all_nodes[0]: np.array([0.0, 0.0])}
    else:
        perplexity = min(30, len(all_nodes) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_2d = tsne.fit_transform(X)
        pos_2d = dict(zip(all_nodes, X_2d))
    # End: Generated via ChatGPT (Prompt: How to reduce transe to 2 dim and apply to networkx graph)

    # Start: Generated via ChatGPT (Prompt: Given a networkx graph and a list of node positions,
    # plot nodes in those positions and display edges with corresponding labels)
    plt.figure(figsize=(12, 8))
    color_map = {"actor": "blue", "director": "green", "movie": "red"}
    for node in all_nodes:
        node_type = graph.nodes[node].get("type", "unknown")
        plt.scatter(*pos_2d[node], c=color_map.get(node_type, "gray"),
                    s=100 if node in node_list else 40)
        plt.text(*pos_2d[node], node, fontsize=8, ha='center', va='bottom')

    # Draw edges
    for u, v in subgraph.edges():
        plt.plot([pos_2d[u][0], pos_2d[v][0]], [pos_2d[u][1], pos_2d[v][1]], 'gray', alpha=0.5)

    # Determine edge labels based on types
    edge_labels = {}
    for u, v in subgraph.edges():
        u_type = graph.nodes[u].get("type")
        v_type = graph.nodes[v].get("type")
        edge_data = graph.get_edge_data(u, v)
        if ("movie" in (u_type, v_type)) and (u_type != v_type):
            # actor/director ↔ movie
            label = edge_data.get("role", "")
        else:
            # works-well-with
            label = edge_data.get("label", "")
        if label:
            edge_labels[(u, v)] = label

    # Draw edge labels
    nx.draw_networkx_edge_labels(subgraph, pos_2d, edge_labels=edge_labels, font_size=8, label_pos=0.5)
    plt.title(f"{graph.graph.get('name', 'graph')} - Nodes: {', '.join(node_list)}")
    plt.axis('off')
    plt.tight_layout()
    filename = f"images/{graph.graph.get('name', 'graph')}_{'_'.join(node_list)}.jpg"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")
    # End: Generated via ChatGPT (Prompt: Given a networkx graph and a list of node positions,

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Visualize: use python kg_visualize.py <kg_name> <node_id1> <node_id2> ...")
        sys.exit(1)

    G_name = sys.argv[1]  # Example: 'kg_base' (file path: kg/kg_base.json)
    node_list = sys.argv[2:] # Example: "Inception (2010)" "Quentin Tarantino" ...

    with open("kg/" + G_name + ".json") as f:
        data = json.load(f)
        G = json_graph.node_link_graph(data)

    generate_node_visualization(G, node_list)
