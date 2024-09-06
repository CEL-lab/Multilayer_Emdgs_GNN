import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import add_self_loops

class EMGNN(torch.nn.Module):
    def __init__(self, nfeat, hidden_channels, n_layers, nclass, meta_x=None, args=None, adj_layers=None, node2idx=None, device=None):
        super().__init__()

        self.args = args
        self.device = device 
        self.linear = nn.Linear(nfeat, hidden_channels)
        self.meta_linear = nn.Linear(nfeat, hidden_channels)

        if args.gcn:
            self.meta_gnn = GCNConv(hidden_channels, hidden_channels)
        elif args.gat:
            self.meta_gnn = GATConv(hidden_channels, hidden_channels, heads=args.nb_heads, concat=False)
        elif args.gin:
            self.meta_gnn = GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.LeakyReLU(), 
                nn.BatchNorm1d(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels)
            ))

        self.classifier = nn.Linear(hidden_channels, nclass)
        self.dropout = args.dropout
        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.n_layers = n_layers

        # Initialize the GCN/GAT/GIN layers
        self.conv = nn.ModuleList()
        for i in range(n_layers):
            if args.gcn:
                self.conv.append(GCNConv(hidden_channels, hidden_channels))
            elif args.gat:
                self.conv.append(GATConv(hidden_channels, hidden_channels, heads=args.nb_heads, concat=False))
            elif args.gin:
                self.conv.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(hidden_channels),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LeakyReLU()
                )))

        # Load multiple network layers as tensors
        self.meta_edge_indices = []
        for adj in adj_layers:  
            meta_edge_index, _ = add_self_loops(adj)
            meta_edge_index = meta_edge_index.to(self.device).long()
            self.meta_edge_indices.append(meta_edge_index.to(self.device)) 

        self.meta_x = meta_x.to(self.device) 

    def forward(self, x, edge_index, meta_edge_index=None, explain_x=None, captum=False, explain=False, edge_weight=None):
        #print("In EMGNN.forward:")
        #print(f"meta_edge_index received type: {type(meta_edge_index)}")        
        if isinstance(meta_edge_index, torch.Tensor):
            print(f"meta_edge_index received shape: {meta_edge_index.shape}")
            print(f"meta_edge_index contents: {meta_edge_index}")
    
        if meta_edge_index is None:
            if len(self.meta_edge_indices) > 0:
                meta_edge_index = self.meta_edge_indices[0]
            else:
                raise ValueError("meta_edge_index must be provided or initialized.")
        
        if not isinstance(meta_edge_index, torch.Tensor) or meta_edge_index.shape[0] != 2:
            raise ValueError(f"meta_edge_index must be a Tensor, but got {type(meta_edge_index)}.")

        if meta_edge_index.dim() != 2 or meta_edge_index.shape[0] != 2:
            print(f"Received meta_edge_index with shape: {meta_edge_index.shape}")
            raise ValueError("meta_edge_index must be a Tensor of shape [2, num_edges].")

        if captum and meta_edge_index is not None:
            meta_x = x[self.nb_nodes:].to(self.device)
            x = x[:self.nb_nodes].to(self.device)

        number_of_nodes = x.shape[0]

        if explain:
            meta_x = x[-1].unsqueeze(dim=0).to(self.device)
            x = x[:-1].to(self.device)

            x = self.leakyrelu(self.linear(x))
            meta_x = self.leakyrelu(self.meta_linear(meta_x))

            for i in range(1):
                x = self.conv[i](x, edge_index.to(self.device))
                x = self.leakyrelu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        
            # Meta message-passing
            x = self.meta_gnn(torch.cat((x, meta_x), dim=0), meta_edge_index)
            x = self.leakyrelu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)
    
        x = self.leakyrelu(self.linear(x.to(self.device)))
        meta_x = self.leakyrelu(self.meta_linear(self.meta_x))

        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index.to(self.device))
            x = self.leakyrelu(x)
            x = F.dropout(x, self.dropout, training=self.training)
    
        # Meta message-passing
        x = self.meta_gnn(torch.cat((x, meta_x), dim=0), meta_edge_index)
        x = self.leakyrelu(x)
        x = F.dropout(x, self.dropout, training=self.training)
    
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, nfeat, hidden_channels, n_layers, nclass, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(nfeat, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, 1))

        self.convs = torch.nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(GCNConv(hidden_channels,hidden_channels))


        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index,edge_weight)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
        #return x.softmax(dim=-1) 

 
class MLP(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,nclass,alpha=0.2):
        super().__init__()
        self.alpha = alpha  
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat)
        self.linear_2 = nn.Linear(outfeat,outd_1)
        self.linear_3 = nn.Linear(outd_1,nclass)

    def forward(self,x,edge_index=None,data=None):
        x = self.leakyrelu(self.linear(x))

        x = F.dropout(x,training=self.training)
        
        x = self.leakyrelu(self.linear_2(x))
        x = F.dropout(x, training=self.training)

        
        x= self.linear_3(x)
        return F.log_softmax(x, dim=1)
 

