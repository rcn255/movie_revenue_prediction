import sys
import os
import json
from networkx.readwrite import json_graph
import networkx as nx
from collections import defaultdict
from itertools import combinations
from two_kg_creation import save_kg

def add_edge_weigts(G):
    # Addin weights to edges based on movie revenue
    for movie_node, data in G.nodes(data=True):
        if data.get('type') != 'movie':
            continue

        movie_revenue = data.get('revenue')
        if not isinstance(movie_revenue, (int, float)) or movie_revenue <= 0:
            continue

        # Distributing revenue as edge weight to connected actors/directors
        for neighbor in G.neighbors(movie_node):
            if G.nodes[neighbor]['type'] in {'actor', 'director'}:
                G[movie_node][neighbor]['weight'] = movie_revenue
                G[neighbor][movie_node]['weight'] = movie_revenue

    return G

def delete_edge_weights(G):
    # Deleting edge weights (only used for pagerank)
    for u, v, attrs in G.edges(data=True):
        if 'weight' in attrs:
            del G[u][v]['weight']

    return G

def pagerank_score_actors(G, alpha=0.9):
    G = add_edge_weigts(G)
    pagerank_scores = nx.pagerank(G, weight='weight', alpha=alpha)

    actor_scores = {
    node: score for node, score in pagerank_scores.items() if G.nodes[node].get('type') == 'actor'
    }

    # Showing top 10 actors
    top_actors = sorted(actor_scores.items(), key=lambda x: x[1], reverse=True)
    for name, score in top_actors[:10]:
        print(f"{name}: {score:.6f}")

    # Giving each actor their pagerank score as attribute
    for node, score in pagerank_scores.items():
        if G.nodes[node].get('type') == 'actor':
            G.nodes[node]['pagerank'] = score

    return G

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Step 4: use python four_kg_pagerank.py <kg_name> <alpha>")
        sys.exit(1)

    G_name = sys.argv[1]  # example: 'kg_base' (will be saved in the 'kg' folder)
    alpha = float(sys.argv[2])  # alpha for pagerank, example=0.9

    with open("kg/" + G_name + ".json") as f:
        data = json.load(f)
        G = json_graph.node_link_graph(data)

    print("Step 4: starting")
    G = pagerank_score_actors(G, alpha)

    save_kg(G, G_name + "_pagerank")
    print("Step 4: completed")
