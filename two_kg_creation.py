import sys
import os
import pandas as pd
import json
import networkx as nx
import ast
from networkx.readwrite import json_graph

def create_kg(df):
    # Columns to exclude from the movie nodes
    exclude_cols = {'cast', 'crew'}

    G = nx.Graph()
    processed = 0 # To track the number of processing movies

    for _, row in df.iterrows():
        try:
            # Parsing the JSON objects
            cast = ast.literal_eval(row['cast'])
            crew = ast.literal_eval(row['crew'])
        except Exception as e:
            continue

        # Since some movies share names, release year is used as part of the key, example: "Inception (2010)"
        release_year = str(row['release_date'])[:4]
        movie_key = f"{row['title']} ({release_year})"

        movie_attributes = {k: v for k, v in row.items() if k not in exclude_cols}
        movie_attributes['type'] = 'movie'

        G.add_node(movie_key, **movie_attributes)

        # Extracting top 5 actors, in the order in which they are mentioned
        for actor in cast[:5]:
            actor_name = actor.get('name')
            #if actor_name:
            G.add_node(actor_name, type='actor', name=actor_name)
            G.add_edge(actor_name, movie_key, role='acted')

        # Directors extracted by defining crew with department 'Directing'
        for person in crew:
            if person.get('department') == 'Directing':
                director_name = person.get('name')
                #if director_name:
                G.add_node(director_name, type='director', name=director_name)
                G.add_edge(director_name, movie_key, role='directed')

        processed += 1


    return G

def save_kg(G, name):
    save_path = os.path.join(os.getcwd(), "kg")
    os.makedirs(save_path, exist_ok=True)

    data = json_graph.node_link_data(G)
    with open(save_path + "/" + name + ".json", "w") as f:
        json.dump(data, f)

    print(f"Finished saving kg to" + save_path + "/" + name + ".json")



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Step 2: use python two_kg_creation.py <df_name> <kg_name>")
        sys.exit(1)

    df_name = sys.argv[1]  # example: 'cleaned_df' (must be present in the 'df' folder)
    kg_name = sys.argv[2]  # example: 'kg_base' (will be saved in the 'kg' folder)

    print("Step 2: starting")
    df = pd.read_csv("df/" + df_name + ".csv")
    G = create_kg(df)

    save_kg(G, kg_name)
    print("Step 2: completed")
