from five_gnn import HeteroGNN
from torch_geometric.data import HeteroData
import torch
import sys
import json
from networkx.readwrite import json_graph
from kg_visualize import generate_node_visualization

def build_subgraph(movie_features_dict, actor_names, director_names, G,
                   person_map, person_feats, model,
                   language_to_idx, genre_to_idx,
                   numeric_mean, numeric_std):

    data = HeteroData()

    # Create person node features (actors + directors)
    people = list(set(actor_names + director_names))
    person_name_to_local_idx = {name: i for i, name in enumerate(people)}
    local_person_feats = torch.stack([person_feats[person_name_to_local_idx[name]] for name in people])
    data['person'].x = local_person_feats

    # Normalizing numeric features using mean and std of all movies in the KG
    runtime = float(movie_features_dict.get('runtime', 0))
    budget = float(movie_features_dict.get('budget', 0))
    numeric_feats = torch.tensor([runtime, budget], dtype=torch.float)
    numeric_feats = (numeric_feats - numeric_mean) / numeric_std.clamp(min=1e-8)

    # Multi-hot encode languages
    lang_vec = torch.zeros(len(language_to_idx))
    for lang in movie_features_dict.get('spoken_languages', []):
        if lang in language_to_idx:
            lang_vec[language_to_idx[lang]] = 1.0

    # Multi-hot encode genres
    genre_vec = torch.zeros(len(genre_to_idx))
    for genre in movie_features_dict.get('genres', []):
        if genre in genre_to_idx:
            genre_vec[genre_to_idx[genre]] = 1.0

    # Concatenate all movie features
    movie_feat = torch.cat([numeric_feats, lang_vec, genre_vec])
    data['movie'].x = movie_feat.unsqueeze(0) # Unsqueezing to get 1D tensor

    # Edge: acted_in
    acted_src = [person_name_to_local_idx[name] for name in actor_names]
    acted_dst = [0] * len(acted_src)
    data['person', 'acted_in', 'movie'].edge_index = torch.tensor([acted_src, acted_dst], dtype=torch.long)

    # Edge: directed
    directed_src = [person_name_to_local_idx[name] for name in director_names]
    directed_dst = [0] * len(directed_src)
    data['person', 'directed', 'movie'].edge_index = torch.tensor([directed_src, directed_dst], dtype=torch.long)

    # Edge: works-well-with
    www_src, www_dst = [], []
    for i, name_i in enumerate(people):
        for j, name_j in enumerate(people):
            if i < j and G.has_edge(name_i, name_j):
                edge_data = G.get_edge_data(name_i, name_j)
                if any(d.get('role') == 'works-well-with' for d in ([edge_data] if isinstance(edge_data, dict) else edge_data.values())):
                    www_src.extend([i, j])
                    www_dst.extend([j, i])
    if www_src:
        data['person', 'works-well-with', 'person'].edge_index = torch.tensor([www_src, www_dst], dtype=torch.long)
    else:
        data['person', 'works-well-with', 'person'].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data

def predict_movie_revenue(subgraph_data, model):
    model.eval()
    subgraph_data = subgraph_data.to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(subgraph_data.x_dict, subgraph_data.edge_index_dict)
        predicted_revenue = out.item()  # only one movie node

    return predicted_revenue

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Step 6: use python six_use_gnn.py <gnn_params> <query_params>")
        sys.exit(1)

    G_name = sys.argv[1]  # example: 'kg_base_logic_pagerank' (from 'kg' folder)
    params_name = sys.argv[2] # example: gnn_params
    params_query = sys.argv[3] # example: query_params

    # Loading the KG
    with open("kg/" + G_name + ".json") as f:
        data = json.load(f)
        G = json_graph.node_link_graph(data)

    # Loading the query
    with open(params_query + ".json", "r") as f:
        data_loaded = json.load(f)
    movie_dict = data_loaded['movie']
    actors = data_loaded['actors']
    directors = data_loaded['directors']

    # Loading mean and stf for budget and runtime
    with open("models/numeric_stats.json", "r") as f:
        stats = json.load(f)

    numeric_mean = torch.tensor(stats["mean"])
    numeric_std = torch.tensor(stats["std"])
    person_map = stats["person_map"]
    person_features = torch.tensor(stats["person_features"])
    language_to_idx = stats["language_to_idx"]
    genre_to_idx = stats["genre_to_idx"]

    # Loading the model
    with open("params/" + params_name + ".json", 'r') as f:
        params = json.load(f)

    model = HeteroGNN(hidden_channels=params["hidden_channels"], out_channels=1, num_layers=params["num_layers"])
    model.load_state_dict(torch.load("models/" + params["model_name"] + ".pt"))

    subgraph = build_subgraph(
        movie_dict,
        actors,
        directors,
        G,
        person_map,
        person_features,
        model,
        language_to_idx,
        genre_to_idx,
        numeric_mean,
        numeric_std
    )

    predicted = predict_movie_revenue(subgraph, model)
    print(f"Predicted revenue: ${predicted}")

    generate_node_visualization(G, actors + directors + ['movie'])
