"""
Dataset columns used: User ID, Follower Count, Retweet Count, Mention Count

Edge Logic :
  Since raw follower/following network data is not in this dataset,
  edges are built using BEHAVIOURAL heuristics:
    1. Users with high Mention Count → interact with top-follower accounts
    2. Users with high Retweet Count → retweet popular accounts
  This is a simplified approximation inspired by interaction behaviour,
  used to simulate a social graph when real network data is unavailable.
"""

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add all users as nodes 
    for _, row in df.iterrows():
        G.add_node(
            row['User ID'],
            username=str(row['Username']),
            followers=int(row['Follower Count']),
            verified=bool(row['Verified']),
            bot=int(row['Bot Label'])
        )

    # Edge Strategy 1: Mention-based interactions 
    # High-mention users likely interact with popular (high-follower) accounts
    top_followed = df.nlargest(50, 'Follower Count')['User ID'].tolist()
    high_mention = df[df['Mention Count'] > df['Mention Count'].median()]

    for _, row in high_mention.iterrows():
        for target in top_followed[:5]:
            if row['User ID'] != target:
                G.add_edge(row['User ID'], target)

    # Edge Strategy 2: Retweet-based interactions 
    # High-retweet users likely retweet from popular accounts
    high_retweet = df[df['Retweet Count'] > df['Retweet Count'].median()]
    for _, row in high_retweet.iterrows():
        for target in top_followed[:3]:
            if row['User ID'] != target:
                G.add_edge(row['User ID'], target)

    print(f"[graph_build] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G
