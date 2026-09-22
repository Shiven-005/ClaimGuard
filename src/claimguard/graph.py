"""
Graph Feature Extractor for ClaimGuard using NetworkX.
Builds an entity graph connecting claimants via shared identifiers (phone, address, bank_account, garage).
Computes graph topological metrics (connected component size, degrees) to flag ring fraud.
"""

import networkx as nx
import pandas as pd

def build_entity_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected entity graph connecting claimants to their shared identifiers.
    """
    G = nx.Graph()

    for idx, row in df.iterrows():
        claimant = f"CLMNT_{row['claimant_id']}"
        phone = f"PHONE_{row['phone']}"
        address = f"ADDR_{row['address']}"
        bank = f"BANK_{row['bank_account']}"
        garage = f"GARAGE_{row['garage_or_hospital_id']}"

        G.add_node(claimant, node_type='claimant')
        G.add_node(phone, node_type='phone')
        G.add_node(address, node_type='address')
        G.add_node(bank, node_type='bank')
        G.add_node(garage, node_type='garage')

        G.add_edge(claimant, phone)
        G.add_edge(claimant, address)
        G.add_edge(claimant, bank)
        G.add_edge(claimant, garage)

    return G

def extract_graph_features(df: pd.DataFrame, G: nx.Graph = None) -> pd.DataFrame:
    """
    Extract graph topological features for each claim in df.
    """
    if G is None:
        G = build_entity_graph(df)

    # Pre-compute connected components
    components = list(nx.connected_components(G))
    node_to_component_size = {}
    node_to_comp_id = {}

    for comp_id, comp in enumerate(components):
        size = len([n for n in comp if n.startswith('CLMNT_')])
        for node in comp:
            node_to_component_size[node] = size
            node_to_comp_id[node] = comp_id

    features = []
    for idx, row in df.iterrows():
        claimant = f"CLMNT_{row['claimant_id']}"
        phone = f"PHONE_{row['phone']}"
        address = f"ADDR_{row['address']}"
        bank = f"BANK_{row['bank_account']}"
        garage = f"GARAGE_{row['garage_or_hospital_id']}"

        deg_phone = G.degree(phone) - 1 if phone in G else 0 # claimants sharing phone
        deg_address = G.degree(address) - 1 if address in G else 0
        deg_bank = G.degree(bank) - 1 if bank in G else 0
        deg_garage = G.degree(garage) - 1 if garage in G else 0

        comp_size = node_to_component_size.get(claimant, 1)
        max_shared = max(deg_phone, deg_address, deg_bank)

        features.append({
            'graph_component_size': comp_size,
            'graph_deg_phone': max(0, deg_phone),
            'graph_deg_address': max(0, deg_address),
            'graph_deg_bank': max(0, deg_bank),
            'graph_deg_garage': max(0, deg_garage),
            'graph_max_shared_count': max(0, max_shared)
        })

    return pd.DataFrame(features, index=df.index)

if __name__ == '__main__':
    from claimguard.synth import generate_synthetic_claims
    print("Testing graph feature extraction...")
    sample_df = generate_synthetic_claims(1000, seed=42)
    graph_feats = extract_graph_features(sample_df)
    print("Extracted graph features head:")
    print(graph_feats.head())
