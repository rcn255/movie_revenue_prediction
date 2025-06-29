python zero_download_datasets.py
python one_preprocessing.py
python two_kg_creation.py combined_df base_kg
python three_kg_logical.py base_kg 7
python four_kg_pagerank.py base_kg_logic 0.9
python five_gnn.py base_kg_logic_pagerank gnn_params
