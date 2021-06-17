---
title: 'Study Notes: GNN Part (I) 🌲 Graph Theory & PyG Usage'
date: 2021-06-15
permalink: /posts/2021/06/gnn-1/
collection: posts
categories:
  - Computer Science
tags:
  - GNN
---

Introduce some fairly straightforward definitions & theorems in graph theory. Include the types of graph, a brief overview of challenges faced by GNNs and an example of basic `PyG` usage. 

Credit to 🔥 Datawhale China; View the GitHub repository [here](https://github.com/datawhalechina/team-learning-nlp/tree/master/GNN) (only Chinese version is available for now.) 

![](/assets/img/banner-1.png)

## Key Points on Graph Theory

(**Adjacency Matrix**) For a graph $\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$, define its adjacency matrix to be $A\in\{0,1\}^{N\times N}$, where $A_{ij} = 1$ implies that there exists an edge from $v_i$ to $v_j$, and does not otherwise.

$\to$ It's easy to observe that the adjacency matrix of an undirected graph is symmetrical.

$\to$ For a weighted graph, the adjacency matrix usually takes value from the weights of its edges.

(**A Seemingly Useful Theorem**) For a graph with an adjacency matrix $A$, the number of walks from $v_i$ to $v_j$ is equal to $A^n_{ij}$ ($i, j^{\text{th}}$-entry of the matrix $A^n$).

*Proof* (Pending)

(**Connected Component**) For a subgraph $\mathcal{G}'=\{\mathcal{V}', \mathcal{E}'\}$ of graph $\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$, if there exists at least one path between any pair of vertices in $\mathcal{V}'$, but does not exist a path connecting any vertex from $\mathcal{V}'$ and any vertex from $\mathcal{V}'\backslash V$, then $\mathcal{G}'$ is said to be a connected component of $\mathcal{G}$.

(**Laplacian Matrix**) For a graph with an adjacency matrix $A$, define its Laplacian matrix to be $\mathbf{L}=\mathbf{D}-\mathbf{A}$, where $\mathbf{D}=\text{diag}(d(v_1), \dots, d(v_N))$, $d:\mathcal{V}\rightarrow\mathbb{N}$ measures the degree of a vertex. Its symmetric normalized version: 

$$
\mathbf{L}=\mathbf{D}^{-\frac{1}{2}}(\mathbf{D}-\mathbf{A})\mathbf{D}^{-\frac{1}{2}}=\mathbf{I}-\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}.
$$

(**Some Fancy Max & Min**)

💙 Shortest path between $v_s$ and $v_t$

$$
p_{st}=\text{arg min}_{p\in\mathcal{P}_{st}}\vert p\vert
$$

, where $\mathcal{P}_{st}$ denotes the set of all paths between the two vertices.

💙 Diameter 

$$
\text{diameter}(\mathcal{G})=\max _{v_s, v_t \in \mathcal{V}} \min _{p \in \mathcal{P}_{st}}\vert p\vert.
$$

## Types of Graph

- Homogeneous $\to$ 1 type of nodes + 1 type of edges
- Heterogeneous $\to$ multiple types of nodes / multiple types of edges
- Bipartite $\to$ 2 types of nodes; edge only exists between nodes of different types

## Machine Learning on Graphs

- **Node Prediction** e.g. classification, regression
- **Edge Prediction** e.g. knowledge graph completion, recommendation system
- **Graph Prediction** e.g. classification, deduction on property
- **Node Clustering** e.g. social circle
- Graph Generation e.g. discover drugs
- Graph Evolution e.g. physics simulation

## Challenges with Graph Structures

Unlike images or texts, which are the main focuses of traditional ML, graphs are irregular and not very structured. Hence, algorithms developed should consider all information available (including the graph's topological structure) and be applicable to vertices of different degrees.

## PyG Assignment

```python
from torch_geometric.data import Data
import torch


class Paper(Data):
    """An object modeling a single graph for some Institute-Author-Paper network,
    inherited from torch.geometric.data.Data:

    Args:
        x_institute (Tensor): Node (of type "institute") feature matrix with shape 
            :obj:`[num_nodes, num_node_features]`. 
        x_author (Tensor): Node (of type "author") feature matrix with shape 
            :obj:`[num_nodes, num_node_features]`. 
        x_paper (Tensor): Node (of type "paper") feature matrix with shape 
            :obj:`[num_nodes, num_node_features]`. 
        edge_author_institute (Tensor): Graph connectivity (of type "Author-Institute" 
            in COO format with shape :obj:`[2, num_edges]`. 
        edge_author_paper (Tensor): Graph connectivity (of type "Author-Paper" 
            in COO format with shape :obj:`[2, num_edges]`. 
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
    """
    def __init__(self, x_institute, x_author, x_paper, edge_author_institute, edge_author_paper, y=None):
        super().__init__(
            x_institute=x_institute, 
            x_author=x_author, 
            x_paper=x_paper,
            edge_author_institute=edge_author_institute,
            edge_author_paper=edge_author_paper,
            y=y
        )

    def find_institute_num(self):
        return self.x_institute.shape[0]

    def find_author_num(self):
        return self.x_author.shape[0]

    def find_paper_num(self):
        return self.x_paper.shape[0]


x_institute = torch.rand(5, 6)
x_author = torch.rand(7, 8)
x_paper = torch.rand(9, 10)
edge_author_institute = torch.randint(1, 6, (2, 10))
edge_author_paper = torch.randint(1, 8, (2, 15))
y = torch.rand(30, 30)

data = Paper(x_institute, x_author, x_paper, edge_author_institute, edge_author_paper, y)
print(data)
print("No.Nodes (Institute):", data.find_institute_num())
print("No.Nodes (Author):", data.find_author_num())
print("No.Nodes (Paper):", data.find_paper_num())
```

