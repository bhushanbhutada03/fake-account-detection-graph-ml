"""
Pipeline:
  1. Load dataset
  2. Build social interaction graph
  3. Extract graph + profile + engineered features
  4. [MAIN]        Graph-based detection (PageRank + degree scoring)
  5. [MAIN]        Community structure analysis
  6. [ENHANCEMENT] ML models
  7. Visualize graph
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, classification_report

from graph_build import build_graph
from feature_engineering import extract_features
from model import train_models


def graph_based_detection(df):
    print("\n" + "="*60)
    print("  STEP 1: GRAPH-BASED DETECTION (MAIN)")
    print("="*60)

    # Use follower_retweet_ratio directly as graph-informed score
    # Low follower + high retweet = bot pattern
    # We combine with pagerank: real users have higher pagerank
    df['graph_score'] = (
        df['follower_retweet_ratio'] * 0.5 +
        df['pagerank'] * 10000 * 0.3 -
        df['mention_retweet_ratio'] * 0.2
    )

    threshold = df['graph_score'].median()
    # High score = real user, Low score = bot
    df['graph_pred'] = (df['graph_score'] < threshold).astype(int)

    acc = accuracy_score(df['Bot Label'], df['graph_pred'])
    print(f"\nGraph Score Threshold : {threshold:.4f}")
    print(f"Graph-based Accuracy  : {acc:.4f}")
    print("\nClassification Report (Graph Only):")
    print(classification_report(df['Bot Label'], df['graph_pred'],
                                target_names=['Real User', 'Bot'],
                                zero_division=0))
    return df


def community_analysis(G):
    print("\n" + "="*60)
    print("  STEP 2: COMMUNITY STRUCTURE ANALYSIS (MAIN)")
    print("="*60)

    communities = list(nx.connected_components(G.to_undirected()))
    sizes = sorted([len(c) for c in communities], reverse=True)

    print(f"\nTotal Communities Found : {len(communities)}")
    print(f"Largest Community Size  : {sizes[0]}")
    print(f"Isolated Nodes (size=1) : {sum(1 for s in sizes if s == 1)}")
    print("\nNote: Isolated nodes are often bot accounts with no real interactions.")
    print("      Dense small clusters may indicate coordinated bot networks.")


def visualize_graph(G, df, sample_size=120):
    print("\n[viz] Drawing graph...")

    bot_set = set(df[df['Bot Label'] == 1]['User ID'].tolist())
    sample_nodes = list(G.nodes())[:sample_size]
    subgraph = G.subgraph(sample_nodes)

    colors = ['#e74c3c' if n in bot_set else '#2ecc71' for n in subgraph.nodes()]
    sizes  = [40 if n in bot_set else 25 for n in subgraph.nodes()]

    plt.figure(figsize=(13, 8))
    pos = nx.spring_layout(subgraph, seed=42, k=0.6)
    nx.draw_networkx_nodes(subgraph, pos, node_color=colors, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, arrows=True,
                           arrowsize=8, edge_color='gray')

    bot_patch  = mpatches.Patch(color='#e74c3c', label='Bot Account')
    real_patch = mpatches.Patch(color='#2ecc71', label='Real User')
    plt.legend(handles=[bot_patch, real_patch], fontsize=12, loc='upper left')
    plt.title('Social Network Graph — Bot vs Real User Detection',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=150)
    plt.show()
    print("[viz] Saved → graph_visualization.png")


def main():
    print("="*60)
    print("  FAKE ACCOUNT DETECTION — BDA MINI PROJECT")
    print("  Graph Analytics (Main) + ML (Enhancement)")
    print("="*60)

    df = pd.read_csv("data/dataset.csv")
    print(f"\n[data] {len(df)} records | Bots: {df['Bot Label'].sum()} | Real: {(df['Bot Label']==0).sum()}")

    G  = build_graph(df)
    df = extract_features(G, df)
    df = graph_based_detection(df)
    community_analysis(G)

    print("\n" + "="*60)
    print("  STEP 3: ML ENHANCEMENT")
    print("="*60)
    print("(ML improves accuracy beyond pure graph detection)\n")
    train_models(df)

    visualize_graph(G, df)
    print("\n✅ Done! Outputs: feature_importance.png | graph_visualization.png")


if __name__ == "__main__":
    main()