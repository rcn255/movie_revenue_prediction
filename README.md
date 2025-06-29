Everything has been tested on a clean Python 3.8.10 env.

Get Started:
`pip install -r requirements.txt`

On Linux, you can run `sh run_pipeline.sh` to run first 5 steps at once. Otherwise run them separately, in which case I recommend running the parameters in the examples.


# Step 0: Downloading data
Example: `python zero_download_datasets.py`

# Step 1: Downloading data
Example: `python one_preprocessing.py`

# Step 2: Knowledge graph creation
`two_kg_creation.py <df_name> <kg_name>`
- <df_name> : dataframe from "df" folder
- <kg_name> : created knowledge graph (KG) to be saved in "kg" folder
Example: `python two_kg_creation.py combined_df base_kg`

# Step 3: Application of logical knowledge on KG
`three_kg_logical.py <kg_name> <IMDb rating>`
- <kg_name> : knowledge graph (KG) stored in "kg" folder
- <IMDb rating> : IMDb rating [0,10] to be used for determine capable actors
Example: `python three_kg_logical.py base_kg 7`

# Step 4: Application of PageRank to score actors
`four_kg_pagerank.py <kg_name> <alpha>`
- <kg_name> : knowledge graph (KG) stored in "kg" folder
- <alpha> : transportation parameter [0,1] for pagerank
Example: `python four_kg_pagerank.py base_kg_logic 0.9`

# Step 5: Downloading data
`five_gnn.py <kg_name>`
- <kg_name> : knowledge graph (KG) stored in "kg" folder
- model and training parameters defined in "params/gnn_params.json"
Example: `python five_gnn.py base_kg_logic_pagerank gnn_params`

# Step 6: Downloading data
`six_use_gnn.py <gnn_params> <query_params>`
- <gnn_params> : model parameters in "params/gnn_params.json"
- <query_params> : query graph structure with movie features, actors and directors (must exist in the original KG)
Example: `python3 six_use_gnn.py base_kg_logic_pagerank gnn_params query_params`

# Utils
`kg_merge.py <kg_name_1> <kg_name_2> <kg_name_combined>`
- merges two knowledge graphs together
- <kg_name_1> : First knowledge graph
- <kg_name_2> : Second knowledge graph
- <kg_name_combined> : Merge of both knowledge graphs

`kg_visualize.py <kg_name> <node_id1> <node_id2> ...`
- visualizes the subgraph containing the specified nodes and their neighbors
- <kg_name> : Base knowledge graph (to get edges from)
- <node_idx> : The x-th node to be visualized along with its neighbors
