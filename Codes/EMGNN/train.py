from __future__ import division
from __future__ import print_function
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.metrics import f1_score
import networkx as nx
from model import EMGNN, MLP
import sys
from torch_geometric.utils import degree
import gcnIO
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from captum.attr import Saliency, IntegratedGradients
from captum_custom import to_captum, Explainer
import itertools
from torch_geometric.utils import add_self_loops, to_undirected, dropout_adj
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('-e', '--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions for GAT.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=250, help='Patience')
parser.add_argument('--n_layers', type=int, default=3, help='Number of Layers')
parser.add_argument('--random_features', type=int, default=0, help='Add random features ')
parser.add_argument('--one_features', type=int, default=0, help='Add identical features')
parser.add_argument('--add_structural_noise', type=float, default=0, help='Remove edges from the graphs')

# Explaining 
parser.add_argument('--pretrained_path', default=None, type=str, help="Path for loading pretrained model")
parser.add_argument('--explain', default=False)
parser.add_argument('--edge_explain', default=False)
parser.add_argument('--node_explain', default=False)
parser.add_argument('--node_edge_explain', default=False)

# Run MLP as baseline instead of EMGNN
parser.add_argument('--mlp', default=False)

# Which GNN to use in the EMGNN
parser.add_argument('--gcn', default=False)
parser.add_argument('--gin', default=False)
parser.add_argument('--gat', default=False)

# Dataset
parser.add_argument('-dataset', '--dataset',
                    help='Input Networks',
                    nargs='+',
                    dest='dataset',
                    default=["IREF_2015", "IREF", "STRING", "PCNET", "MULTINET", "CPDB"])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def find_model_name(args):
    if args.gcn:
        return "GCN"
    elif args.gin:
        return "GIN"
    elif args.gat:
        return "GAT"
    #elif args.goat:
     #   return "GOAT"
    elif args.mlp:
        return "MLP"
    else:
        print("No model selected. Use --gcn True or --gat True to select the gnn model for EMGNN or --mlp True for the baseline MLP")
        exit()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    print("Number of predicted positive", preds.sum(), "/", preds.shape[0])
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):
    t = time.time()
    model.train()
    data = next(iter(loader)).to(device) 
    optimizer.zero_grad()
    
    if args.mlp:
        output = model(meta_x.float().to(device)).squeeze()
    else:
        #meta_edge_index = model.meta_edge_indices[0] if hasattr(model, 'meta_edge_indices') else None
        #output = model(data.x.float().to(device), data.edge_index.to(device), data).squeeze()
        meta_edge_index = model.meta_edge_indices[0] if hasattr(model, 'meta_edge_indices') else None
        output = model(data.x.float().to(device), data.edge_index.to(device), meta_edge_index).squeeze()
        output = output[number_of_input_nodes:]

    #loss_train = loss(output[idx_train], meta_y[idx_train])
    print("Shape of output[idx_train]:", output[idx_train].shape)
    print("Shape of meta_y[idx_train]:", meta_y[idx_train].shape)
    #loss_train = loss(output[idx_train], meta_y[idx_train].squeeze())
    loss_train = loss(output[idx_train], meta_y[idx_train].view(-1))

    acc_train = accuracy(output[idx_train], meta_y[idx_train])
    loss_train.backward()
    optimizer.step()

    print(f'Epoch: {epoch+1}, loss_train: {loss_train.item():.4f}, acc_train: {acc_train.item():.4f}, time: {time.time() - t:.4f}s')

    return loss_train.item()

def compute_test(write_results=True, accs=[]):
    model.eval()
    data = next(iter(loader)).to(device)

    if args.mlp:
        output = model(meta_x.float().to(device)).squeeze()
    else:
        meta_edge_index = model.meta_edge_indices[0] if hasattr(model, 'meta_edge_indices') else None
        print(f"meta_edge_index in compute_test() received type: {type(meta_edge_index)}")

        output = model(data.x.float().to(device), data.edge_index.to(device), meta_edge_index).squeeze()
        output = output[number_of_input_nodes:]

    loss_test = loss(output[idx_test], meta_y[idx_test].view(-1))
    acc_test = accuracy(output[idx_test], meta_y[idx_test].view(-1))

    auroc = roc_auc_score(
        y_score=torch.exp(output[idx_test, 1]).cpu().detach().numpy(),
        y_true=meta_y[idx_test].cpu().detach().numpy()
    )
    aupr = average_precision_score(
        y_score=torch.exp(output[idx_test, 1]).cpu().detach().numpy(),
        y_true=meta_y[idx_test].cpu().detach().numpy()
    )

    print("aupr", aupr)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()),
          "rocauc={:.4f}".format(auroc),
          "aupr={:.4f}".format(aupr))

    accs.append(acc_test.item())
    max_acc = max(accs)

    if write_results:
        if not os.path.exists("./results"):
            os.makedirs("./results")
            
        with open("./results/results.txt", "a") as handle:
            if args.random_features == 1 or args.one_features == 1:
                handle.write(f'{args.dataset} {type(model).__name__} metagnn:{find_model_name(args)} "random_features":{args.random_features} "one_features":{args.one_features}  n_layers:{args.n_layers} hidden:{args.hidden} loss_test:{str(loss_test.item())} acc_test:{str(acc_test.item())} ,aupr:{aupr} auroc:{auroc}' )
            elif args.add_structural_noise > 0:
                handle.write(f'{args.dataset} {type(model).__name__} metagnn:{find_model_name(args)}  "structural_noise":{args.add_structural_noise}  n_layers:{args.n_layers} hidden:{args.hidden} loss_test:{str(loss_test.item())} acc_test:{str(acc_test.item())} ,aupr:{aupr} auroc:{auroc}' )
            else:
                handle.write(f'{args.dataset} {type(model).__name__} metagnn:{find_model_name(args)} n_layers:{args.n_layers} hidden:{args.hidden} loss_test:{str(loss_test.item())} acc_test:{str(max_acc)} ,aupr:{aupr} auroc:{auroc}' )
            handle.write("\n")

    return acc_test.item(), accs[-1], aupr, auroc, torch.exp(output).cpu().detach().numpy()  # convert output to prob

# Start
find_model_name(args)
paths = []
paths.append("/mmfs1/home/muhammad.kazim/EMGNN/container_with_node_names_R1.h5")

data_list = []
node2idx = {}  # init node2idx dict to find common nodes among graphs.
counter = 0
train_nodes = []
test_nodes = []
val_nodes = []
meta_y = torch.zeros(100000000, 1)  # init with some default big value
y_list = []
node_names_list = []
number_nodes_each_graph = []

features_order = [
    'AM Notes', 'Area SAIDI', 'Ark Grid Mod or OK Grid Enhancement Circuits', 'CAD_ID',
    'CMI', 'CMI Category', 'Call Code', 'Call Qty', 'Cause Desc', 'Cause ID',
    'Dev Subtype', 'Device Address', 'Device Type', 'Equip Desc', 'Equip ID',
    'Equipment Desc that should be excluded from reported indices', 'Event Exclusion?',
    'Exclude CMI?', 'Extent', 'Feeder ID', 'Feeder SAIDI', 'Job ASAI', 'Job CAIDI',
    'Job City', 'Job Display ID', 'Job Duration Mins', 'Job Feeder', 'Job OFF Time',
    'Job ON Time', 'Job QA?', 'Job SAIDI', 'Job SAIFI', 'Lead Crew', 'Lead Crew Phone',
    'Region SAIDI', 'STRCTUR_NO/Job Device ID', 'Subst SAIDI', 'Total Area Premises',
    'Total Feeder Premises', 'Total Region Premises', 'Total Subst Premises',
    'Total System Premises', 'Transmission Voltage (69kV, 138kV, 161kv) feeding distribution substation', 'Year'
]

# Initialize node2idx and counter
node2idx = {}
counter = 0

# Iterate over each path
for path in paths:
    data = gcnIO.load_hdf_data(path, feature_name='node_features')
    adj_layers, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data

    # Ensure node_names are properly formatted strings
    if isinstance(node_names[0], bytes):
        node_names = [i.decode("utf-8").strip() for i in node_names]  # Decode and strip
    elif isinstance(node_names[0], tuple):
        node_names = [''.join(i).strip() for i in node_names]  # Join tuples into strings and strip

    #print("Formatted node_names:", node_names[:10])  # Print first 10 for inspection
    #print("node2idx keys:", list(node2idx.keys())[:10])  # Print first 10 keys for inspection

    # Construct node2idx dict and increment counter
    for name in node_names:
        node_str = name.strip().lower()  # Ensure the node_str is formatted correctly
        #print(f"Storing node: '{node_str}'")
        if node_str not in node2idx:
            node2idx[node_str] = counter
            counter += 1

    # Initialize meta_x after building node2idx
    if 'meta_x' not in globals():
        meta_x = torch.zeros((len(node2idx), features.shape[1]))  # Correct initialization with proper dimensions
        meta_y = torch.zeros(len(node2idx), dtype=torch.long)  # Initialize meta_y with zeros

    #y_train = y_train.astype(int)
    #y_test = y_test.astype(int)

    y_train = (y_train > 0).astype(int)
    y_test = (y_test > 0).astype(int) 

    #y = torch.cat([torch.tensor(y_train), torch.tensor(y_test)], dim=0)
    #y_list.append(y)

    y = torch.cat([torch.tensor(y_train), torch.tensor(y_test)], dim=0)
    y_list.append(y)
    
    for i, label in enumerate(y):
        node_str = node_names[i].strip().lower()
        #print(f"Accessing node2idx with key: '{node_str}'")
        if node_str in node2idx:
            idx = node2idx[node_str]
            if meta_y[idx] == 0:
                meta_y[idx] = label
    # Diagnostic print for keys in node2idx after processing
    #print("Final node2idx keys:", list(node2idx.keys())[:10])
    #print("Final meta_x shape:", meta_x.shape)

    idx_train = [i for i, x in enumerate(train_mask) if x == 1]
    idx_test = [i for i, x in enumerate(test_mask) if x == 1]

    adj_tensors = [torch.FloatTensor(layer) for layer in adj_layers]
    adj = adj_tensors
    
# Early check for out-of-bounds indices in adjacency matrices
    for i, adj_tensor in enumerate(adj_tensors):
        #print(f"Layer {i+1} adj_tensor contents:\n{adj_tensor}")
        edge_index = adj_tensor.long()
        # Directly use adj_tensor as the edge_index
        edge_index = adj_tensor.long()
        #print(f"Layer {i+1} edge_index contents before processing:\n{edge_index}")
    # Checking if edge_index contains any out-of-bounds indices
        max_index = edge_index.max().item()
        if max_index >= len(node_names):
            print(f"Out-of-bounds index found in layer {i+1}. Maximum index: {max_index}")
            sys.exit()

    #if len(y_train) != np.sum(train_mask):
     #   raise ValueError(f"Mismatch: train_mask length {np.sum(train_mask)} does not match y_train length {len(y_train)}")

    y_train = torch.LongTensor(y_train[train_mask.astype(np.bool_)]).squeeze()
    y_test = torch.LongTensor(y_test[test_mask.astype(np.bool_)]).squeeze()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    train_nodes_batch = [node_names[idx.item()] for idx in idx_train]
    test_nodes_batch = [node_names[idx.item()] for idx in idx_test]

    train_nodes.append(train_nodes_batch)
    test_nodes.append(test_nodes_batch)

    node_names_list.append(node_names)

    num_positives = torch.sum(y_train, dim=0)
    num_negatives = len(y_train) - num_positives
    pos_weight = num_negatives / num_positives

    edge_indices = []

    for i, adj_tensor in enumerate(adj_tensors):
        edge_index = adj_tensor.long()
        max_adj_index = adj_tensor.max().item()
        # Check if any adjacency matrix references out-of-bounds indices
        if max_adj_index >= len(node_names):
            raise ValueError(f"Layer F**K {i+1}: Adjacency matrix has out-of-bounds index {max_adj_index}. Maximum allowed index is {len(node_names) - 1}.")

        # Convert the feature matrix to a tensor
        features_tensor = torch.tensor(features, dtype=torch.float)
        #print(f"Layer {i+1} features_tensor shape: {features_tensor.shape}")
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_names))
        # Check for out-of-bounds indices before adding self-loops
        max_index = edge_index.max().item()
        if max_index >= len(node_names):
            #print(f"edge_index: {edge_index}")
            raise ValueError(f"Found an out-of-bounds index in edge_index: {max_index}")

        data = Data(x=features_tensor, edge_index=edge_index, y=y, node_names=node_names)
        data_list.append(data)

# Ensure all node names are strings before any lookup
    node_names = [name.strip() for name in node_names]

    #print("Final node2idx keys:", list(node2idx.keys())[:10])
    #print("Final meta_x shape:", meta_x.shape)
    #print("Final meta_y shape:", meta_y.shape)

if node_names_list:
    node_names = np.concatenate(node_names_list, axis=0)
else:
    print("node_names_list is empty, cannot concatenate.")

meta_x = meta_x[:len(node2idx)]
number_nodes_each_graph.append(meta_x.shape[0])

node_names = np.concatenate(node_names_list, axis=0)

meta_node_names = np.zeros(len(node2idx), dtype="object")

for k, v in node2idx.items():
    if isinstance(k, tuple) and len(k) == 2:
        k1, k2 = k 
        meta_node_names[v] = k2
    else:
        print(f"Unexpected key MF *** structure: {k}")

# meta_y was previously populated with labels for meta nodes
#meta_y = torch.tensor(meta_y[:len(node2idx)]).type(torch.LongTensor).squeeze()
print("Unique values in meta_y:", torch.unique(meta_y))
meta_y = torch.tensor(meta_y[:len(node2idx)]).long().squeeze()

# Ensure that meta_y has the same number of dimensions as y_list elements
if meta_y.dim() == 1:
    meta_y = meta_y.unsqueeze(1)  # Ensure it has the same dimension as y_list elements

# Add this debug print
#print(f"meta_y sample: {meta_y[:10]}")

# Concatenate all y elements from y_list
y = torch.concat(y_list, dim=0)

# Ensure y is also 2D before concatenation
if y.dim() == 1:
    y = y.unsqueeze(1)  # Ensure y is 2D

# Finally, concatenate y and meta_y
y = torch.concat([y, meta_y], dim=0)

train_set = {node.strip().lower() for t in train_nodes for node in t}
t2 = {node.strip().lower() for t in val_nodes for node in t}
t3 = {node.strip().lower() for t in test_nodes[:-1] for node in t}
train_set = train_set.union(t2)
train_set = train_set.union(t3)

val_set = set()
for i, val in enumerate(itertools.islice(train_set, int(len(train_set)*0.1))):
    val_set.add(val)

test_set = {node.strip().lower() for node in test_nodes[-1]}

train_set = train_set - train_set.intersection(test_set)
val_set = val_set - val_set.intersection(test_set)
train_set = train_set - val_set.intersection(val_set)
try:
    print(f"train_set sample: {list(train_set)[:10]}")
    idx_train = torch.tensor([node2idx[i] for i in train_set])
except KeyError as e:
    print(f"KeyError: The key '{e.args[0]}' was not found in node2idx. Potential issue with key formatting.")
    raise

idx_test = torch.tensor([node2idx[i] for i in test_set])
idx_val = torch.tensor([node2idx[i] for i in val_set])

if args.random_features == 1:
    for data in data_list:
        for i, node in enumerate(data.node_names):
            idx = node2idx[tuple(node)]    
            torch.manual_seed(idx)
            rv = torch.rand(data.x.shape[1])
            data.x[i, :] = rv
            meta_x[idx] = rv
elif args.one_features == 1:
    for data in data_list:
        data.x = torch.ones((data.x.shape[0], data.x.shape[1]))
    meta_x = torch.ones((meta_x.shape[0], meta_x.shape[1]))

num_positives = torch.sum(meta_y[idx_train], dim=0)
num_negatives = len(meta_y[idx_train]) - num_positives
pos_weight = num_negatives / num_positives

#print("Data types in data_list:", [type(data) for data in data_list])  # Debug print to check types in data_list

loader = DataLoader(data_list, batch_size=len(data_list))

batch = next(iter(loader))

# Add these debug prints
#print(f"Batch type: {type(batch)}")  # Check if batch is a Data object
#print(f"Type of batch.x: {type(batch.x)}")
#print(f"Contents of batch.x: {batch.x}")

number_of_input_nodes = batch.x.shape[0]
if args.cuda:
    meta_y = meta_y.squeeze().to(device)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    idx_val = idx_val.to(device)
    pos_weight = pos_weight.to(device)
    batch = batch.to(device)
    meta_x = meta_x.to(device)

if args.mlp:
    model = MLP(batch.x.shape[1], args.hidden, args.hidden, 2)
else:
    model = EMGNN(nfeat=batch.x.shape[1], 
                hidden_channels=args.hidden,
                n_layers=args.n_layers,
              nclass=2,
              meta_x=meta_x,
              args=args,
              adj_layers=adj_tensors,  # Assuming adj_tensors corresponds to adj_layers
              node2idx=node2idx,
              device=device)


if args.cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

loss = F.nll_loss

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
accs = []

for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

    if (epoch % 50 == 0):
        print("test results epoch:", epoch)
        accs.append(compute_test(write_results=False)[0])

print("evaluation on last epoch")
accs.append(compute_test(write_results=False)[0])

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

test_performance = compute_test(write_results=True, accs=accs)

if args.pretrained_path:
    pretrained_network = args.pretrained_path.split("/")[-2].split("_")[1]
    model_dir = f"{find_model_name(args)}_{args.dataset}_pretrained_{pretrained_network}"
else:
    model_dir = f"{find_model_name(args)}_{args.dataset}"
model_dir = gcnIO.create_model_dir(model_dir)

gcnIO.save_predictions(model_dir, np.array(list(node2idx.keys())), test_performance[4])
gcnIO.write_hyper_params2(args, os.path.join(model_dir, 'hyper_params.txt'))

torch.save(model.state_dict(), '{}/model.pkl'.format(model_dir))

idx2node = {v: k for k, v in node2idx.items()}

with open(f"{model_dir}/idx_train.pkl", "wb") as handle:
    pickle.dump(idx_train, handle)
with open(f"{model_dir}/idx_val.pkl", "wb") as handle:
    pickle.dump(idx_val, handle)
with open(f"{model_dir}/idx_test.pkl", "wb") as handle:
    pickle.dump(idx_test, handle)

with open(f"{model_dir}/args.pkl", "wb") as handle:
    pickle.dump(args, handle)

with open(f"{model_dir}/batch.pkl", "wb") as handle:
    pickle.dump(batch.cpu(), handle)

with open(f"{model_dir}/meta_x.pkl", "wb") as handle:
    pickle.dump(meta_x.cpu(), handle)

with open(f"{model_dir}/node2idx.pkl", "wb") as handle:
    pickle.dump(node2idx, handle)

with open(f"{model_dir}/idx2node.pkl", "wb") as handle:
    pickle.dump(idx2node, handle)

with open(f"{model_dir}/meta_edge_index.pkl", "wb") as handle:
    pickle.dump(model.meta_edge_index.cpu(), handle)

with open(f"{model_dir}/final_y.pkl", "wb") as handle:
    pickle.dump(y.cpu(), handle)

all_node_names = np.concatenate([node_names[:, 1], meta_node_names], axis=0)
args.dataset.append("Meta_Node")
g = 0
for i, n in enumerate(all_node_names):
    if i < number_nodes_each_graph[g]:
        all_node_names[i] = n + "_" + args.dataset[g]
    else:
        g += 1
        number_nodes_each_graph[g] += number_nodes_each_graph[g-1]
        all_node_names[i] = n + "_" + args.dataset[g]

with open(f"{model_dir}/all_node_names.pkl", "wb") as handle:
    pickle.dump(all_node_names, handle)

with open(f"{model_dir}/edge_index.pkl", "wb") as handle:
    pickle.dump(batch.edge_index.cpu(), handle)

final_edge_index = torch.concat([batch.edge_index.cuda(), model.meta_edge_index.cuda()], dim=1).cuda()
with open(f"{model_dir}/final_edge_index.pkl", "wb") as handle:
    pickle.dump(final_edge_index.cpu(), handle)
