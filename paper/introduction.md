# Introduction

Financial fraud costs the global economy hundreds of billions of dollars annually. As payment systems grow more digital and interconnected, fraudsters exploit relationship structures — sharing devices, IP addresses, or merchants — to carry out coordinated attacks that evade traditional detection methods.

Conventional fraud detection relies on supervised machine learning: Logistic Regression, Random Forest, XGBoost, and MLP models classify each transaction using tabular features such as amount, time, location, and merchant category. These models capture individual transaction signals but are blind to the relational context: a single transaction appears normal in isolation but becomes suspicious when connected to a network of fraudulent accounts.

Graph Neural Networks (GNNs) address this limitation by propagating information across edges in a transaction graph, enabling each node to aggregate signals from its neighbourhood. A fraudulent account connected to many other flagged accounts will receive a reinforced fraud signal even if its individual features are unremarkable.

This paper makes the following contributions:

1. A complete graph-based fraud detection framework (GraphShield) using PyTorch Geometric.
2. Graph construction from the Elliptic Bitcoin benchmark dataset.
3. Training and comparison of three GNN architectures: GCN, GraphSAGE, and GAT.
4. Comparative evaluation against five ML baselines.
5. Explainability analysis using GNNExplainer to surface fraud-relevant subgraphs.
6. An ablation study isolating the contribution of graph structure and class balancing.
