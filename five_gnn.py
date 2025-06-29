import json
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader

from sklearn.model_selection import train_test_split
from networkx.readwrite import json_graph
import networkx as nx
import numpy as np

import math

import matplotlib.pyplot as plt

class HeteroGNN(nn.Module): # https://pytorch-geometric.readthedocs.io/en/2.5.1/notes/heterogeneous.html
    def __init__(self, hidden_channels=64, out_channels=1, num_layers=2):
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('person', 'acted_in', 'movie'): SAGEConv((-1, -1), hidden_channels),
                ('person', 'directed', 'movie'): SAGEConv((-1, -1), hidden_channels),
                ('person', 'works-well-with', 'person'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: x.relu() for k, x in x_dict.items()}
        return self.lin(x_dict['movie']).squeeze()  # Predict revenue for movies

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data['movie'].train_mask], data['movie'].y[data['movie'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(): # For eval and test
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    losses = {}
    for split in ['train', 'val', 'test']:
        mask = data['movie'][f'{split}_mask']
        loss = criterion(out[mask], data['movie'].y[mask])
        losses[split] = loss.item()
    return losses


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Step 5: use python five_gnn.py <kg_name>")
        sys.exit(1)

    G_name = sys.argv[1]  # example: 'kg_base' (from 'kg' folder)
    params_name = sys.argv[2] # example: gnn_params

    with open("kg/" + G_name + ".json") as f:
        data = json.load(f)
        G = json_graph.node_link_graph(data)

    with open("params/" + params_name + ".json", 'r') as f:
        params = json.load(f)

    # 1. Edge tensors
    print(f"Step 5: 1. Creating edge tensors")
    actor_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'actor']
    director_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'director']
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'movie']

    # Combining actors and directors together
    person_nodes = actor_nodes + director_nodes
    person_map = {name: i for i, name in enumerate(person_nodes)} # Name to index
    movie_map = {name: i for i, name in enumerate(movie_nodes)} # Name to index

    # Encoding person features (actor + director combined)
    person_features = []
    for n in person_nodes:
        d = G.nodes[n]
        capable = float(1) if d.get('capable', False) else float(0)
        pagerank = float(d.get('pagerank', 0))
        person_features.append(torch.tensor([capable, pagerank], dtype=torch.float))
    person_features = torch.stack(person_features)

    # Building edges in terms of indices
    acted_src, acted_dst = [], []
    directed_src, directed_dst = [], []

    # Works-well-with edges
    www_src, www_dst = [], []

    for u, v, d in G.edges(data=True):
        role = d.get('role')
        label = d.get('label')
        u_type = G.nodes[u].get('type')
        v_type = G.nodes[v].get('type')

        if role == 'acted' and u_type == 'actor' and v_type == 'movie':
            acted_src.append(person_map[u])
            acted_dst.append(movie_map[v])
        elif role == 'directed' and u_type == 'director' and v_type == 'movie':
            directed_src.append(person_map[u])
            directed_dst.append(movie_map[v])
        elif label == 'works-well-with':
            # Only add if both ends are person (actor or director)
            if u_type in {'actor', 'director'} and v_type in {'actor', 'director'}:
                www_src.append(person_map[u])
                www_dst.append(person_map[v])

    acted_edge_index = torch.tensor([acted_src, acted_dst], dtype=torch.long) if acted_src else torch.empty((2,0), dtype=torch.long)
    directed_edge_index = torch.tensor([directed_src, directed_dst], dtype=torch.long) if directed_src else torch.empty((2,0), dtype=torch.long)
    www_edge_index = torch.tensor([www_src, www_dst], dtype=torch.long) if www_src else torch.empty((2,0), dtype=torch.long)

    # 2. Feature tensors
    print(f"Step 5: 2. Creating feature tensors")
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'movie']
    movie_map = {n: i for i, n in enumerate(movie_nodes)}

    # Build vocabularies for categorical multi-hot features
    all_languages = set()
    all_genres = set()
    for n in movie_nodes:
        d = G.nodes[n]
        all_languages.update(d.get('spoken_languages', []))
        all_genres.update(d.get('genres', []))

    language_to_idx = {lang: i for i, lang in enumerate(sorted(all_languages))}
    genre_to_idx = {genre: i for i, genre in enumerate(sorted(all_genres))}
    print(f"All languages: {all_languages}, {len(all_languages)}")
    print(f"All genres: {all_genres}, {len(all_genres)}")

    numeric_feature_list = []
    categorical_feature_list = []
    movie_targets = []

    # Collecting numerical features
    for n in movie_nodes:
        d = G.nodes[n]
        runtime = float(d.get('runtime', 0))
        budget = float(d.get('budget', 0))
        numeric_feature_list.append([runtime, budget])

    numeric_tensor = torch.tensor(numeric_feature_list, dtype=torch.float)

    # Normalizing numerical features (runtime, budget)
    numeric_mean = numeric_tensor.mean(dim=0)
    numeric_std = numeric_tensor.std(dim=0)
    numeric_tensor = (numeric_tensor - numeric_mean) / numeric_std.clamp(min=1e-8) # Handling div by 0

    # Saving numeric_mean and numeric_std for normalization of queries
    stats = {
        "mean": numeric_mean.tolist(),
        "std": numeric_std.tolist(),
        "person_map": person_map,
        "person_features": person_features.tolist(),
        "language_to_idx": language_to_idx,
        "genre_to_idx": genre_to_idx,
        }

    with open("models/numeric_stats.json", "w") as f:
        json.dump(stats, f)
    
    # Combined feature vectors (normalized numeric + one-hot)
    movie_features = []
    for i, n in enumerate(movie_nodes):
        d = G.nodes[n]

        lang_vec = torch.zeros(len(language_to_idx))
        for lang in d.get('spoken_languages', []):
            if lang in language_to_idx:
                lang_vec[language_to_idx[lang]] = 1.0

        genre_vec = torch.zeros(len(genre_to_idx))
        for genre in d.get('genres', []):
            if genre in genre_to_idx:
                genre_vec[genre_to_idx[genre]] = 1.0

        feat = torch.cat([numeric_tensor[i], lang_vec, genre_vec])
        movie_features.append(feat)

        # Target: revenue
        revenue = float(d.get('revenue', 0))
        movie_targets.append(revenue)

    movie_features = torch.stack(movie_features)
    movie_targets = torch.tensor(movie_targets, dtype=torch.float)
    print(f"Movie features shape: {movie_features.shape}")
    print(f"Movie targets shape: {movie_targets.shape}")

    # 3. Creating HeteroData as outlined in https://pytorch-geometric.readthedocs.io/en/2.5.1/notes/heterogeneous.html
    print(f"Step 5: 3. Creating HeteroData object")
    data = HeteroData()

    data['movie'].x = movie_features
    data['movie'].y = movie_targets

    data['person'].x = person_features

    data['person', 'acted_in', 'movie'].edge_index = acted_edge_index
    data['person', 'directed', 'movie'].edge_index = directed_edge_index

    data['person', 'works-well-with', 'person'].edge_index = www_edge_index

    num_movies = data['movie'].num_nodes
    indices = torch.randperm(num_movies)

    # 80-10-10 split
    train_idx = indices[: int(0.8 * num_movies)]
    val_idx = indices[int(0.8 * num_movies) : int(0.9 * num_movies)]
    test_idx = indices[int(0.9 * num_movies):]

    # Store masks for convenience
    data['movie'].train_mask = torch.zeros(num_movies, dtype=torch.bool)
    data['movie'].val_mask = torch.zeros(num_movies, dtype=torch.bool)
    data['movie'].test_mask = torch.zeros(num_movies, dtype=torch.bool)

    data['movie'].train_mask[train_idx] = True
    data['movie'].val_mask[val_idx] = True
    data['movie'].test_mask[test_idx] = True

    # 4. Initiating model and defining parameters
    print(f"Step 5: 4. Initiating model and defining parameters")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # If CUDA is available
    model = HeteroGNN(hidden_channels=params["hidden_channels"], out_channels=1, num_layers=params["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.MSELoss()
    data = data.to(device)

    best_val_rmse = float('inf')
    early_stop = 0
    train_rmse_history = []
    val_rmse_history = []
    early_stopping_patience = params['early_stopping_patience']
    model_path = 'models/'

    # 5. Starting training
    print(f"Step 5: 5. Starting training")

    for epoch in range(1, params['max_epochs'] + 1):
        loss = train()
        losses = test()
        train_rmse = math.sqrt(losses['train'])
        val_rmse = math.sqrt(losses['val'])

        train_rmse_history.append(train_rmse)
        val_rmse_history.append(val_rmse)

        print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        # Saving best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            early_stop = 0
            torch.save(model.state_dict(), model_path + params['model_name'] + ".pt")
            print(f"New best model saved (Val RMSE: {val_rmse:.4f})")
        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f"Early stopping at {epoch}")
                break

    # Loading best model
    model.load_state_dict(torch.load(model_path + params['model_name'] + ".pt"))

    # Test only after training completes
    final_losses = test()
    test_rmse = math.sqrt(final_losses['test'])
    print(f"Step 5: Final Test RMSE: {test_rmse:.4f}")

    # Plot RMSE over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_history, label='Train RMSE')
    plt.plot(val_rmse_history, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/rmse_plot.png")
    plt.show()

    print("Step 5: finished")
