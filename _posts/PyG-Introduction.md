---
layout: post
title: PyTorch Geometric Tutorial 0: Introduction
---

**NOTICE!** This is a Chinese tutorial translated from [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) for study use. Please visit their website for more information.

## 下载并引入库
```
# Install required packages.
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Helper function for visualization.
%matplotlib inline
import torch
import networkx as nxc
import matplotlib.pyplot as plt


def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
```

## 图神经网络 (Graph Neural Network) 介绍
图神经网络旨在泛化经典的深度学习概念至非常规的数据结构 (与图像或文字不同), 使得神经网络能够对研究对象与其关系进行推理. 举个例子, 我们可以使用一个简单的 **神经消息传递架构 (neural message passing scheme)** 来实现这一需求. 在某个图中 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, 所有节点 $v \in \mathcal{V}$ 的特征向量 $\mathbf{x}_v^{(\ell)}$ 都会被从它们的邻居 (neighbors) $\mathcal{N}(v)$ 聚集 (aggregate) 本地信息, 来进行迭代地更新:

$$
\mathbf{x}_v^{(\ell + 1)} = f^{(\ell + 1)}_{\theta} \left( \mathbf{x}_v^{(\ell)}, \left\{ \mathbf{x}_w^{(\ell)} : w \in \mathcal{N}(v) \right\} \right)
$$

```
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
```
输出结果:
```
Dataset: KarateClub():
======================
Number of graphs: 1
Number of features: 34
Number of classes: 4
```

在初始化`KarateClub`数据集之后, 我们首先可以检查它的一些属性. 例如, 我们可以看到这个数据集只有一张图, 并且每个节点都有一个34维的特征向量 (用来 **唯一** 描述Karate俱乐部的每个成员). 另外, 这张图只有4个类别, 这代表每个节点所属的社区. 现在让我们看看这个图的更多细节:

```
data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```
输出结果:
```
Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
==============================================================
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Number of training nodes: 4
Training node label rate: 0.12
Has isolated nodes: False
Has self-loops: False
Is undirected: True
```
在PyTorch Geometric内, 每张图都由某个`Data`对象表示, 该对象拥有所有描述图表达 (graph representation) 的信息. 我们可以通过`print(data)`打印数据对象:
```
Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
```
不难发现, 这个`data`对象拥有4个属性: 

(1) 节点特征向量 `x`: 34个节点, 每个节点有一个34维的特征向量

(2) 图连接性 (connectivity) `edge_index`: 每条边, 表示源 (source) 节点到目标 (destination) 节点是否相连

(3) 节点标签 `y`: 每个节点都只有一个类别

(4) 还有一个额外的属性叫`train_mask`, 描述了我们已知节点的社区分配

总的来说, 我们只知道4个节点的真实 (groud-truth) 标签, 我们的任务是推断剩余节点的社区分配.

`data`对象也提供一些便利函数来推断一些图下基础属性. 比如, 我们可以轻易判断在图中是否存在孤立节点, 也就是其与任意其他节点都没有边相连. 又或者是这个图是否包含自循环 (self-loop), 数学上我们可以表示为 $(v, v) \in \mathcal{E}$. 还有是否这是个无向图, 也就是对于每条边 $(v, w) \in \mathcal{E}$, 同样存在$(w, v) \in \mathcal{E}$. 让我们接下来检查一下`edge_index`的属性:

```
edge_index = data.edge_index
print(edge_index.t())
```
输出结果:
```
tensor([[ 0,  1],
        [ 0,  2],
        [ 0,  3],
        ...
        [33, 30],
        [33, 31],
        [33, 32]])
```
通过打印出`edge_index`, 我们就可以理解PyG是怎样在内部表达图连接性的了. 我们可以看到对于每条边, `edge_index`都拥有两个索引的元组, 其中第一个值是源节点的索引, 而第二个值则是对应边的目标节点的索引.

这种表达也被叫做COO格式 (coordinate format), 它通常被用作表达稀疏矩阵. 与在密集表达内持有邻接信息 $\mathbf{A} \in {0, 1}^{|\mathcal{V}| \times |\mathcal{V}|}$ 不同, PyG稀疏地表达图, 也就是只持有 $\mathbf{A}$ 里非零的坐标或值. 而且, PyG并不区分有向和无向图, 它将无向图视作有向图的一个特例. 我们可以将这个图转化为 `networkx` 库格式, 来可视化它:

```
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)
```
输出结果: 