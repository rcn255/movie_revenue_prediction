import sys
import os
import json
from networkx.readwrite import json_graph
import networkx as nx
from collections import defaultdict
from itertools import combinations
from two_kg_creation import save_kg

def identifying_capable_actors(G, imdb_rating=7):
    # Adding the "capable" attribute to actors based on the average vote_average of their movies
    # If the average rating of their movies > imdb_rating, then they are "capable"

    for node, data in G.nodes(data=True):
        if data.get('type') == 'actor':
            # Get neighboring movies
            neighbor_movies = [
                nbr for nbr in G.neighbors(node)
                if G.nodes[nbr].get('type') == 'movie'
            ]

            # Skiping actors with no movies
            if not neighbor_movies:
                continue

            ratings = [
                G.nodes[movie].get('vote_average')
                for movie in neighbor_movies
                if isinstance(G.nodes[movie].get('vote_average'), (int, float))
            ]

            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                if avg_rating > imdb_rating:
                    G.nodes[node]['capable'] = True
                else:
                    G.nodes[node]['capable'] = False

    return G

def identifying_work_well_together(G, imdb_rating=7):
    # Tracking how often pairs of actors appear working on a movie together
    pair_counts = defaultdict(int)

    # Collect co-appearances from movies rated > imdb_rating
    for node, data in G.nodes(data=True):
        if data.get('type') == 'movie' and data.get('vote_average', 0) >= 7:
            neighbors = [
                nbr for nbr in G.neighbors(node)
                if G.nodes[nbr].get('type') == 'actor' or G.nodes[nbr].get('type') == 'director'
            ]

            # Counting pair
            for actor1, actor2 in combinations(sorted(neighbors), 2):
                pair_counts[(actor1, actor2)] += 1

    # Adding edges for frequent collaborators
    for (actor1, actor2), count in pair_counts.items():
        if count >= 2:
            G.add_edge(actor1, actor2, label='works-well-with') # Using label instead of role

    return G


def apply_logical_knowledge(G, imdb_rating=7):

    G = identifying_capable_actors(G, imdb_rating)
    print("Step 3: finished identifying capable actors")

    capable_count = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'actor' and d.get('capable'))
    print(f"Step 3: number of capable actors: {capable_count}")

    G = identifying_work_well_together(G, imdb_rating)
    print("Step 3: finished identifying actors that work well together")

    collab_edges = sum(1 for u, v, d in G.edges(data=True) if d.get('label') == 'works-well-with')
    print(f"Step 3: number of 'works-well-with' edges added: {collab_edges}")

    return G

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Step 3: use python three_kg_logical.py <kg_name> <IMDb rating>")
        sys.exit(1)

    G_name = sys.argv[1]  # example: 'kg_base' (will be saved in the 'kg' folder)
    imdb_rating = float(sys.argv[2])  # example: 'kg_base' (will be saved in the 'kg' folder)

    with open("kg/" + G_name + ".json") as f:
        data = json.load(f)
        G = json_graph.node_link_graph(data)

    print("Step 3: starting")
    G = apply_logical_knowledge(G, imdb_rating)
    print(f"Step 3: finished applying logical knowledge")

    save_kg(G, G_name + "_logic")
    print("Step 3: completed")
