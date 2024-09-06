from __future__ import division
from __future__ import print_function

from torch_geometric.data import Data
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
from torch_geometric.data import DataLoader
import sys
from torch_geometric.utils import degree
import gcnIO
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
#from torch_geometric.nn import GNNExplainer
#from gnn_explainer import GNNExplainer
from captum.attr import Saliency, IntegratedGradients
from captum_custom import to_captum,Explainer
import itertools
from torch_geometric.utils import add_self_loops,to_undirected,dropout_adj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('-e','--epochs', type=int, default=2000, help='Number of epochs to train.')
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
parser.add_argument('--add_structural_noise', type=float, default= 0, help='Remove edges from the graps')

#explaining 
parser.add_argument('--pretrained_path', default= None, type=str, help="Path for loading pretrained model")
parser.add_argument('--explain', default=False)
parser.add_argument('--edge_explain', default=False)
parser.add_argument('--node_explain', default=False)
parser.add_argument('--node_edge_explain', default=False)

#run mlp as baseline instead of EMGNN
parser.add_argument('--mlp', default=False)

#which gnn to use in the EMGNN
parser.add_argument('--gcn', default=False)
parser.add_argument('--gin', default=False)
parser.add_argument('--gat', default=False)

#dataset
parser.add_argument('-dataset', '--dataset',
                        help='Input Networks',
                        nargs='+',
                        dest='dataset',
                        default=["IREF_2015","IREF", "STRING", "PCNET", "MULTINET", "CPDB" ])


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def find_model_name(args):
    if(args.gcn):
        return "GCN"
    elif(args.gin):
        return "GIN"
    elif(args.gat):
        return "GAT"
    elif(args.goat):
        return "GOAT"
    elif(args.mlp):
        return "MLP"
    else:
        print("No model selected. Use --gcn True or --gat True to select the gnn model for emgnn or --mlp True for the baseline MLP")
        exit()

def accuracy(output, labels):  
    preds = output.max(1)[1].type_as(labels)
    #preds = (output>0).float() #this accuracy when we use BCEWithLogitsLoss with 1 output.
    print("Number of predicted positive",preds.sum(),"/",preds.shape[0])
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):

    t = time.time()
    model.train()
    data = next(iter(loader)).cuda()
    optimizer.zero_grad()
    if(args.mlp):
        output = model(meta_x.float().cuda()).squeeze()
    else:
        output = model(data.x.float().cuda(),data.edge_index.cuda(),data).squeeze()
        output = output[number_of_input_nodes:]
    #g = make_dot(output, model.state_dict())
    #g.view()
    #loss_train = F.nll_loss(output[idx_train], y_train)
    loss_train = loss(output[idx_train], meta_y[idx_train])
    acc_train = accuracy(output[idx_train], meta_y[idx_train])
    loss_train.backward()
    #plot_grad_flow2(model.named_parameters())

    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        if(args.mlp):
            output = model(meta_x.float().cuda()).squeeze()
        else:
            output = model(data.x.float().cuda(),data.edge_index.cuda(),data).squeeze()
            output = output[number_of_input_nodes:]
        
    #loss_val = F.nll_loss(output[idx_val],y_val)
    loss_val = loss(output[idx_val],meta_y[idx_val])
    acc_val = accuracy(output[idx_val], meta_y[idx_val])

    #these if we have one raw output 
    #auroc = roc_auc_score(y_score=torch.sigmoid(output[idx_train]).cpu().detach().numpy(), y_true=meta_y[idx_train].cpu().detach().numpy())
    #aupr = average_precision_score(y_score=torch.sigmoid(output[idx_train]).cpu().detach().numpy(), y_true=meta_y[idx_train].cpu().detach().numpy())

    #these if we have two output units with logsoftmax
    auroc = roc_auc_score(y_score=torch.exp(output[idx_train,1]).cpu().detach().numpy(), y_true=meta_y[idx_train].cpu().detach().numpy())
    aupr = average_precision_score(y_score=torch.exp(output[idx_train,1]).cpu().detach().numpy(), y_true=meta_y[idx_train].cpu().detach().numpy())

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'auroc_train: {:.4f}'.format(auroc),
          'aupr_train: {:.4f}'.format(aupr),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()

def compute_test(write_results=True,accs=[]):
    model.eval()
    data = next(iter(loader)).cuda()
    
    if(args.mlp):
        output = model(meta_x.float().cuda()).squeeze()
    else:
        output = model(data.x.float().cuda(),data.edge_index.cuda(),data).squeeze()
        output = output[number_of_input_nodes:]

    
    #loss_test = F.nll_loss(output[idx_test], y_test)
    loss_test = loss(output[idx_test], meta_y[idx_test])

    acc_test = accuracy(output[idx_test], meta_y[idx_test])

    #auroc = roc_auc_score(y_score=torch.sigmoid(output[idx_test]).cpu().detach().numpy(), y_true=meta_y[idx_test].cpu().detach().numpy())
    #aupr = average_precision_score(y_score=torch.sigmoid(output[idx_test]).cpu().detach().numpy(), y_true=meta_y[idx_test].cpu().detach().numpy())

    auroc = roc_auc_score(y_score=torch.exp(output[idx_test,1]).cpu().detach().numpy(), y_true=meta_y[idx_test].cpu().detach().numpy())
    aupr = average_precision_score(y_score=torch.exp(output[idx_test,1]).cpu().detach().numpy(), y_true=meta_y[idx_test].cpu().detach().numpy())
    print("aupr",aupr)
    
    #fpr, tpr, _ = roc_curve(y_score=output[idx_test].cpu().detach().numpy(), y_true=meta_y[idx_test].cpu().detach().numpy()) #for one output unit
    fpr, tpr, _ = roc_curve(y_score=output[idx_test,1].cpu().detach().numpy(), y_true=meta_y[idx_test].cpu().detach().numpy())

    #fig = plt.figure(figsize=(20, 12))
    #plt.plot(fpr, tpr, lw=4, alpha=0.3, label='(AUROC = %0.2f)' % (auroc))
    #fig.savefig(f'./slurm_output/{find_model_name(args)}_roc_curve.pdf')
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()),
          "rocauc={:.4f}".format(auroc),
          "aupr={:.4f}".format(aupr))

    accs.append(acc_test.item())
    max_acc = max(accs)
    if(write_results):
        with open("./results/results.txt","a") as handle:
            if(args.random_features==1 or args.one_features==1):
                handle.write(f'{args.dataset} { type(model).__name__} metagnn:{find_model_name(args)} "random_features":{args.random_features} "one_features":{args.one_features}  n_layers:{args.n_layers} hidden:{args.hidden} loss_test:{str(loss_test.item())} acc_test:{str(acc_test.item())} ,aupr:{aupr} auroc:{auroc}' )
            elif(args.add_structural_noise>0):
                handle.write(f'{args.dataset} { type(model).__name__} metagnn:{find_model_name(args)}  "structural_noise":{args.add_structural_noise}  n_layers:{args.n_layers} hidden:{args.hidden} loss_test:{str(loss_test.item())} acc_test:{str(acc_test.item())} ,aupr:{aupr} auroc:{auroc}' )
            else:
                handle.write(f'{args.dataset} { type(model).__name__} metagnn:{find_model_name(args)} n_layers:{args.n_layers} hidden:{args.hidden} loss_test:{str(loss_test.item())} acc_test:{str(max_acc)} ,aupr:{aupr} auroc:{auroc}' )
            handle.write("\n")
        

    return acc_test.item(),accs[-1],aupr,auroc, torch.exp(output).cpu().detach().numpy() #convert output to prob


#Start
find_model_name(args)
paths = []
paths.append("/mmfs1/home/muhammad.kazim/EMGNN/container.h5")

data_list = []
node2idx = {} #init node2idx dict to find commong nodes ammong graphs.
counter = 0
train_nodes = []
test_nodes = []
val_nodes = []
meta_y = torch.zeros(1000000,1) #init with some default big value
y_list =[]
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
meta_x = torch.zeros((100000,44)) #init with some default big value

for path in paths:
    data = gcnIO.load_hdf_data(path, feature_name='features')
    adj_layers, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data

    feature_names = [f.decode("utf-8") for f in feature_names]

   # Align features across all graphs (layers)
    feature_ind = [feature_names.index(f_n) for f_n in features_order if f_n in feature_names]
    feature_names = np.array(feature_names)[feature_ind]
    features = features[:, feature_ind]

    # Construct node2idx dict -> useful to find common nodes among graphs
    for i in node_names:
        if tuple(i) not in node2idx:
            node2idx[tuple(i)] = counter
            counter += 1

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    if y_val is not None:
        y_val = y_val.astype(int)

    # Initialize y_all with zeros of the same length as node_names
    y_all = np.zeros(len(node_names), dtype=int)

    # Ensure the masks align with node names
    if len(train_mask) != len(node_names):
        raise ValueError(f"train_mask length {len(train_mask)} does not match the number of nodes {len(node_names)}")
    if len(test_mask) != len(node_names):
        raise ValueError(f"test_mask length {len(test_mask)} does not match the number of nodes {len(node_names)}")
    if y_val is not None and len(val_mask) != len(node_names):
        raise ValueError(f"val_mask length {len(val_mask)} does not match the number of nodes {len(node_names)}")

    # Assign labels based on masks
    y_all[train_mask.astype(bool)] = y_train
    y_all[test_mask.astype(bool)] = y_test
    if y_val is not None:
        y_all[val_mask.astype(bool)] = y_val

    # Convert y_all to a torch tensor
    y_all = torch.tensor(y_all)
    y_list.append(y_all)

    print(f"Length of y_all: {len(y_all)}")
    print(f"Length of node_names: {len(node_names)}")
# Process each node and feature
    for i, label in enumerate(y_all):
        try:
            idx = node2idx[tuple(node_names[i])]
            print(f"Processing node {i}, mapped idx: {idx}")
            meta_x[idx] = torch.from_numpy(features[i])
            if meta_y[idx] == 0:
                meta_y[idx] = label
        except IndexError as e:
            print(f"IndexError at i={i}, node_names[i]={node_names[i] if i < len(node_names) else 'N/A'}, len(node_names)={len(node_names)}")
            raise e

  # Process each network layer (adjacency matrix)
    for layer_idx, adj in enumerate(adj_layers):
        adj = torch.FloatTensor(np.array(adj))
        edge_index = (adj > 0).nonzero().t()
        edge_index, _ = add_self_loops(edge_index)

        if args.add_structural_noise > 0:  # Add structural noise if required
            edge_index = dropout_adj(edge_index, p=args.add_structural_noise, force_undirected=True)[0]

        # Create the PyG Data object for the layer
        data = Data(x=features, edge_index=edge_index, y=y_all, node_names=node_names)
        data_list.append(data)

    # Store the node names for concatenation later
    node_names_list.append(node_names)

# Final concatenation of node names after processing all layers
if node_names_list:
    node_names = np.concatenate(node_names_list, axis=0)
else:
    raise ValueError("node_names_list is empty. No arrays to concatenate.")

meta_x = meta_x[:len(node2idx)]
number_nodes_each_graph.append(meta_x.shape[0])

# Process node names for meta nodes
meta_node_names = np.zeros(len(node2idx), dtype="object")

for k, v in node2idx.items():
    if isinstance(k, tuple):
        if len(k) > 1:
            # If the tuple is longer than expected, join it back into a single string
            meta_node_names[v] = ''.join(k)  # Convert tuple back to a string
        else:
            print(f"Unexpected tuple size in node2idx: {k}")
            meta_node_names[v] = k[0]  # Use the first element if it's a single-element tuple
    else:
        meta_node_names[v] = k  # If k is not a tuple, use it directly

# Ensure meta_y is correctly shaped
meta_y = torch.tensor(meta_y[:len(node2idx)]).type(torch.LongTensor)

# Define y by concatenating y_list
y = torch.cat(y_list, dim=0)

# Ensure y is 2D before concatenation
if y.dim() == 1:
    y = y.unsqueeze(1)  # Convert to 2D tensor by adding a dimension

# Ensure meta_y is also 2D
if meta_y.dim() == 1:
    meta_y = meta_y.unsqueeze(1)  # Convert to 2D tensor by adding a dimension

# Now, you can safely concatenate them
y = torch.cat([y, meta_y], dim=0)

train_set = {tuple(node) for t in train_nodes for node in t}  # Remove duplicate nodes across graphs
t2 = {tuple(node) for t in val_nodes for node in t}
t3 = {tuple(node) for t in test_nodes[:-1] for node in t}
train_set = train_set.union(t2).union(t3)

# Add 10% of the training set to the val set
val_set = set(itertools.islice(train_set, int(len(train_set) * 0.1)))

test_set = {tuple(node) for node in test_nodes[-1]}  # Test only in nodes from the last graph

train_set -= train_set.intersection(test_set)
val_set -= val_set.intersection(test_set)
train_set -= val_set.intersection(train_set)

idx_train = torch.tensor([node2idx[i] for i in train_set])
idx_test = torch.tensor([node2idx[i] for i in test_set])
idx_val = torch.tensor([node2idx[i] for i in val_set])

if(args.random_features == 1): #remove node features
    
    for data in data_list: #same random vector for the same genes accross networks
        for i,node in enumerate(data.node_names):
            idx = node2idx[tuple(node)]    
            torch.manual_seed(idx)
            rv = torch.rand(data.x.shape[1])
            data.x[i,:] = rv #same random vector for the same genes accross networks
            meta_x[idx] = rv

elif(args.one_features == 1):
    for data in data_list:
        data.x = torch.ones((data.x.shape[0],data.x.shape[1]))
    meta_x = torch.ones((meta_x.shape[0],meta_x.shape[1]))

#model_dir = "./results/my_models/GCN_['CPDB', 'IREF_2015', 'PCNET', 'STRING', 'MULTINET', 'IREF']_2022_10_05_05_19_17"

num_positives = torch.sum(meta_y[idx_train], dim=0)
num_negatives = len(meta_y[idx_train]) - num_positives
pos_weight  = num_negatives / num_positives


loader = DataLoader(data_list,batch_size=len(data_list))

batch = next(iter(loader))

number_of_input_nodes = batch.x.shape[0]
if args.cuda:
    meta_y = meta_y.squeeze().cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()
    idx_val = idx_val.cuda()
    pos_weight= pos_weight.cuda()
    batch = batch.cuda()
    meta_x= meta_x.cuda()

if(args.mlp):
    model = MLP(batch.x.shape[1],args.hidden,args.hidden,2)
else:
    model = EMGNN(batch.x.shape[1],
                        args.hidden,
                        args.n_layers,
                        nclass=2,
                        args=args,
                        data=batch,
                        meta_x=meta_x,
                        node2idx=node2idx)


if args.cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)



#loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#loss = F.nll_loss(weight=torch.tensor([1,pos_weight]).cuda())
loss = F.nll_loss

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
accs = []

#path_to_model = "./results/my_models/GCN_['IREF', 'IREF_2015', 'STRING', 'PCNET', 'MULTINET', 'CPDB']_2022_12_13_14_23_50/model"
#model.load_state_dict(torch.load('{}.pkl'.format(path_to_model)))
#test_performance = compute_test(write_results=True,accs=accs)

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

    if (epoch % 50 == 0 ):
        print("test results epoch:",epoch)
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

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
# Testing
test_performance = compute_test(write_results=True,accs=accs)

if(args.pretrained_path):
    pretrained_network = args.pretrained_path.split("/")[-2].split("_")[1] 
    model_dir = f"{find_model_name(args)}_{args.dataset}_pretrained_{pretrained_network}"
else:
    model_dir = f"{find_model_name(args)}_{args.dataset}"
model_dir = gcnIO.create_model_dir(model_dir) #this function will also add date to the folder name 
gcnIO.save_predictions(model_dir, np.array(list(node2idx.keys())), test_performance[4])
gcnIO.write_hyper_params2(args, os.path.join(model_dir, 'hyper_params.txt'))

#save files to model directory
torch.save(model.state_dict(), '{}/model.pkl'.format(model_dir))

idx2node = {v: k for k, v in node2idx.items()}

with open(f"{model_dir}/idx_train.pkl","wb") as handle:
    pickle.dump(idx_train,handle)
with open(f"{model_dir}/idx_val.pkl","wb") as handle:
    pickle.dump(idx_val,handle)
with open(f"{model_dir}/idx_test.pkl","wb") as handle:
    pickle.dump(idx_test,handle)
    
with open(f"{model_dir}/args.pkl","wb") as handle:
    pickle.dump(args,handle)

with open(f"{model_dir}/batch.pkl","wb") as handle:
    pickle.dump(batch.cpu(),handle)

with open(f"{model_dir}/meta_x.pkl","wb") as handle:
    pickle.dump(meta_x.cpu(),handle)

with open(f"{model_dir}/node2idx.pkl","wb") as handle:
    pickle.dump(node2idx,handle)

with open(f"{model_dir}/idx2node.pkl","wb") as handle:
    pickle.dump(idx2node,handle)

with open(f"{model_dir}/meta_edge_index.pkl","wb") as handle:
    pickle.dump(model.meta_edge_index.cpu() ,handle)

with open(f"{model_dir}/final_y.pkl","wb") as handle:
    pickle.dump(y.cpu(),handle)

all_node_names = np.concatenate([node_names[:,1],meta_node_names],axis=0)
args.dataset.append("Meta_Node")
g =0
for i,n in enumerate(all_node_names):
    if(i<number_nodes_each_graph[g]):
        all_node_names[i] = n+"_"+args.dataset[g]
    else:
        g+=1
        number_nodes_each_graph[g]+=number_nodes_each_graph[g-1]
        all_node_names[i] = n+"_"+args.dataset[g]

with open(f"{model_dir}/all_node_names.pkl","wb") as handle:
        pickle.dump(all_node_names,handle)

with open(f"{model_dir}/edge_index.pkl","wb") as handle:
    pickle.dump(batch.edge_index.cpu(),handle)

final_edge_index = torch.concat([batch.edge_index.cuda(),model.meta_edge_index.cuda()],dim=1).cuda()
with open(f"{model_dir}/final_edge_index.pkl","wb") as handle:
    pickle.dump(final_edge_index.cpu(),handle)

