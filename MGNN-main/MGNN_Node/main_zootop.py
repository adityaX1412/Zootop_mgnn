import os.path as osp
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from typing import Dict, Tuple
from zootop_dataloader import SNDLibParser
import random
import torch_geometric.transforms as T
from layer import *


class SNDLibDataset:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.content = f.read()
        self.parser = SNDLibParser(self.content)
        self.parser.parse()
        
        # Convert to format compatible with the GNN
        self.num_nodes = len(self.parser.nodes)
        self.num_features = 2  # longitude and latitude
        self.num_classes = len(set(d.routing_unit for d in self.parser.demands.values()))
        
    def process(self) -> Tuple[Data, torch.Tensor]:
        # Create node features (coordinates)
        x = torch.zeros((self.num_nodes, self.num_features))
        for node_id, node in self.parser.nodes.items():
            x[node_id] = torch.tensor([node.longitude, node.latitude])
            
        # Create edge index and edge weights
        edge_index = []
        edge_weights = []
        for link in self.parser.links.values():
            edge_index.append([link.source, link.target])
            # Normalize edge weight based on module capacity
            weight = 1.0 / (link.module_capacity + 1e-6)  
            edge_weights.append(weight)
            
            # Add reverse edge for undirected graph
            edge_index.append([link.target, link.source])
            edge_weights.append(weight)
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Create demand matrix (can be used as labels/targets)
        demand_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        for demand in self.parser.demands.values():
            demand_matrix[demand.source, demand.target] = demand.demand_value
            
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weights,
            num_nodes=self.num_nodes
        )
        
        return data, demand_matrix

def main(file_path: str, seed: int, verbose: bool, num_epochs: int, auto_ml: bool = False):
    print(f"running on seed: {seed}...")
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and process SNDLib data
    dataset = SNDLibDataset(file_path)
    data, demand_matrix = dataset.process()
    data = data.to(device)
    demand_matrix = demand_matrix.to(device)
    
    # Create graph for MotifConv
    edge_index = data.edge_index
    x = data.x
    
    # Create sparse adjacency matrix
    row, col = edge_index.cpu().numpy()
    sp_mat = sp.coo_matrix(
        (np.ones_like(row), (row, col)), 
        shape=(x.shape[0], x.shape[0])
    )
    
    # Normalize adjacency matrix
    edge_weight_norm = normalize_adj(sp_mat).data.reshape([-1, 1])
    
    # Create motif matrices
    mc = MotifCounter('custom', [sp_mat], osp.dirname(file_path))
    motif_mats = mc.split_13motif_adjs()
    motif_mats = [
        convert_sparse_matrix_to_th_sparse_tensor(normalize_adj(motif_mat)).to(device) 
        for motif_mat in motif_mats
    ]
    
    # Setup graph for DGL
    num_filter = 1
    num_edge = edge_index.shape[1]
    weight_index_data = np.array([range(num_filter)], dtype=np.int32).repeat(num_edge, axis=0)
    
    rel_type = [str(rel) for rel in set(weight_index_data.flatten().tolist())]
    graph_data = {('P', rel, 'P'): [[], []] for rel in rel_type}
    edge_data = {rel: [] for rel in rel_type}
    
    for rel in rel_type:
        for eid in range(weight_index_data.shape[0]):
            for j in range(num_filter):
                if str(weight_index_data[eid, j]) == rel:
                    graph_data[('P', rel, 'P')][0].append(row[eid])
                    graph_data[('P', rel, 'P')][1].append(col[eid])
                    edge_data[rel].append([edge_weight_norm[eid, 0]])
    
    graph_data = {rel: tuple(graph_data[rel]) for rel in graph_data}
    g = dgl.heterograph(graph_data).int().to(device)
    
    for rel in rel_type:
        g.edges[rel].data['edge_weight_norm'] = torch.tensor(
            edge_data[rel], 
            dtype=torch.float32
        ).to(device)
    
    # Model parameters
    hidden_dim1 = 16
    compress_dims = [6]
    att_act = torch.sigmoid
    layer_dropout = [0.5, 0.6]
    motif_dropout = 0.1
    att_dropout = 0.1
    mw_initializer = 'Xavier_Uniform'
    kernel_initializer = None
    bias_initializer = None
    
    # Define the network
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = MotifConv(
                dataset.num_features, 
                hidden_dim1,
                compress_dims[0],
                rel_type,
                'custom',
                motif_mats,
                mw_initializer,
                att_act,
                motif_dropout,
                att_dropout,
                aggr='sum'
            )
            
            self.dense = Linear(
                13 * compress_dims[-1],
                dataset.num_classes,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            )
        
        def forward(self, g, h):
            h = F.dropout(h, p=layer_dropout[0], training=self.training)
            h = self.conv1(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=layer_dropout[1], training=self.training)
            h = self.dense(h)
            return F.log_softmax(h, dim=1)
    
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Create train/val/test masks (you may need to adjust this based on your needs)
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(g, x)
        loss = F.nll_loss(out[data.train_mask], demand_matrix[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Evaluation function
    @torch.no_grad()
    def evaluate():
        model.eval()
        out = model(g, x)
        loss_val = F.nll_loss(out[data.val_mask], demand_matrix[data.val_mask])
        loss_test = F.nll_loss(out[data.test_mask], demand_matrix[data.test_mask])
        return loss_val.item(), loss_test.item()
    
    # Training loop
    best_val_loss = float('inf')
    test_loss = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train()
        val_loss, tmp_test_loss = evaluate()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = tmp_test_loss
            
        if verbose and epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    file_path = '/kaggle/input/sndlib/sndlib-networks-native/india35.txt'
    main(file_path, seed=42, verbose=True, num_epochs=1000)
