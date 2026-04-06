"""
Extracts graph-based (structural) + profile-based engineered features.

Graph features  → capture network behaviour (PageRank, clustering)
Profile features → capture account statistics from dataset columns
"""

import networkx as nx
import pandas as pd


def extract_features(G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:

    # Graph-based features
    print("[features] Computing PageRank (alpha=0.85)...")
    pagerank = nx.pagerank(G, alpha=0.85)

    print("[features] Computing in-degree and out-degree...")
    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    print("[features] Computing clustering coefficient...")
    clustering = nx.clustering(G.to_undirected())

    graph_rows = []
    for node in G.nodes():
        graph_rows.append({
            "User ID":    node,
            "pagerank":   pagerank.get(node, 0),
            "in_degree":  in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
            "clustering": clustering.get(node, 0),
        })

    graph_df = pd.DataFrame(graph_rows)
    df = pd.merge(df, graph_df, on="User ID")

    # Profile-based engineered features
    # 1. Follower-to-Retweet ratio: bots get few retweets despite many followers
    df['follower_retweet_ratio'] = df['Follower Count'] / (df['Retweet Count'] + 1)

    # 2. Mention ratio: bots mention many but get fewer real interactions
    df['mention_retweet_ratio'] = df['Mention Count'] / (df['Retweet Count'] + 1)

    # 3. Verified flag as integer
    df['Verified'] = df['Verified'].astype(int)

    # 4. Account age proxy from Created At (older accounts less likely bots)
    try:
        df['Created At'] = pd.to_datetime(df['Created At'])
        ref_date = pd.Timestamp('2024-01-01')
        df['account_age_days'] = (ref_date - df['Created At']).dt.days.clip(lower=0)
    except Exception:
        df['account_age_days'] = 0

    # 5. Has hashtags (1/0)
    df['has_hashtags'] = df['Hashtags'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() in ['', 'NaN', 'nan'] else 1
    )

    print(f"[features] Total features ready: {df.shape[1]} columns")
    return df