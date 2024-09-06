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
parser.add_argument('--hidden', type=int, default=44, help='Number of hidden units.')
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
        output = model(data.x.float().cuda(), edge_indices_list, data).squeeze()
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
paths.append("/mmfs1/home/muhammad.kazim/EMGNN/emgnn_data_with_features_fixed.h5")

#for dataset in args.dataset:
 #   if("CPDB" == dataset):
  #      paths.append("./results/EMOGI_CPDB/CPDB_multiomics.h5")
   # elif("IREF" == dataset):
    #    paths.append("./results/EMOGI_IRefIndex/IREF_multiomics.h5")
    #elif("IREF_2015" == dataset):
     #   paths.append("./results/EMOGI_IRefIndex_2015/IREF_2015_multiomics.h5")
    #elif("MULTINET" == dataset):
     #   paths.append("./results/EMOGI_Multinet/MULTINET_multiomics.h5")
    #elif("PCNET" == dataset):
     #   paths.append("./results/EMOGI_PCNet/PCNET_multiomics.h5")
    #elif("STRING" == dataset):
     #   paths.append("./results/EMOGI_STRINGdb/STRINGdb_multiomics.h5")


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

# Initialize meta_x based on the correct feature size
#meta_x = torch.zeros((len(node2idx), features.shape[1])).cuda()
meta_x = None  # Initialize meta_x as None

for path in paths:
    data = gcnIO.load_hdf_data(path, feature_name='features')
    edge_indices, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data


    print(f"Processing path: {path}")
    print(f"Shapes - y_train: {y_train.shape}, y_val: {y_val.shape if y_val is not None else 'None'}, y_test: {y_test.shape}")
    print(f"Shapes - node_names: {len(node_names)}, features: {features.shape}")
    
    feature_names = [f.decode("utf-8") for f in feature_names]

    # Align features in all the graphs
    feature_ind = [feature_names.index(f_n) for f_n in features_order if f_n in feature_names]
    feature_names = np.array(feature_names)[feature_ind]

    features = features[:, feature_ind]
    number_nodes_each_graph.append(features.shape[0])

    # Initialize meta_x based on the correct feature size during the first loop iteration
    if meta_x is None:
        meta_x = torch.zeros((len(node2idx), features.shape[1]))

    # Construct node2idx dict -> useful to find common nodes among graphs.
    for i in node_names:
        if tuple(i) not in node2idx:
            node2idx[tuple(i)] = counter
            counter += 1

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    if y_val is not None:
        y_val = y_val.astype(int)
    
    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)

    if y_val is not None:
        y_val_tensor = torch.tensor(y_val)
        y = torch.cat([y_train_tensor, y_test_tensor, y_val_tensor], dim=0)
    else:
        y = torch.cat([y_train_tensor, y_test_tensor], dim=0)


    print(f"Length of y: {len(y)}")
    print(f"Length of node_names: {len(node_names)}")

    for i, label in enumerate(y):
        if tuple(node_names[i]) in node2idx:
            idx = node2idx[tuple(node_names[i])]
            print(f"Processing node {i}, mapped idx: {idx}")

            # Resize meta_x if necessary
            if idx >= meta_x.shape[0]:
                meta_x = torch.cat([meta_x, torch.zeros(1, features.shape[1])], dim=0)
            meta_x[idx] = torch.from_numpy(features[i])
            if meta_y[idx] == 0:
                meta_y[idx] = label
        
        else:
            print(f"IndexError at i={i}, node_names[i]=N/A, len(node_names)={len(node_names)}")
            raise IndexError(f"IndexError at i={i}: len(node_names)={len(node_names)}, len(y)={len(y)}")

    # Adjust the train, val, test labels to be in format-> [r,1] instead of [N,1] where r is the number of samples of the set.
    idx_train = [i for i, x in enumerate(train_mask) if x == 1]
    if val_mask is not None:
        idx_val = [i for i, x in enumerate(val_mask) if x == 1]
    else:
        idx_val = []  # or set to None if you'd prefer
    idx_test = [i for i, x in enumerate(test_mask) if x == 1]
    
    features = torch.FloatTensor(features)

    # Do not reapply the mask; use y_train and y_test directly
    y_train = torch.LongTensor(y_train).squeeze()
    if y_val is not None:
        y_val = torch.LongTensor(y_val).squeeze()
    y_test = torch.LongTensor(y_test).squeeze()

    idx_train = torch.LongTensor(idx_train)
    if idx_val:
        idx_val = torch.LongTensor(idx_val)
    else:
        idx_val = None
    idx_test = torch.LongTensor(idx_test)

    # Correctly append node names based on indices
    train_nodes.extend([node_names[i] for i in idx_train.tolist()])
    test_nodes.extend([node_names[i] for i in idx_test.tolist()])
    if idx_val is not None:
        val_nodes.extend([node_names[i] for i in idx_val.tolist()])

    num_positives = torch.sum(y_train, dim=0)
    num_negatives = len(y_train) - num_positives
    pos_weight  = num_negatives / num_positives
# Initialize edge_indices_list as an empty list
    edge_indices_list = []

    for adj in edge_indices:
        print(f"Type of adj: {type(adj)}")  # Print the type of adj
        print(f"Contents of adj: {adj}")  # Print the contents of adj
    
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj)  # Convert numpy array to PyTorch tensor

        if isinstance(adj, torch.Tensor):
        # Proceed only if adj is a tensor
            edge_index = (adj > 0).nonzero().t()
            edge_index, _ = add_self_loops(edge_index)
            edge_indices_list.append(edge_index)
        else:
            print(f"Warning: adj is not a tensor, skipping. Type: {type(adj)}")

    if len(edge_indices_list) > 0:
        data = Data(x=features, edge_index=edge_indices_list[0], y=y, node_names=node_names)
    else:
        raise ValueError("No valid edge indices were found.")

    data = Data(x=features, edge_index=edge_indices_list[0], y=y, node_names=node_names)

    # Appending data to data_list
    data_list.append(data)

    if(args.add_structural_noise>0): #add structural noise
        # num_edges_to_add = int(args.add_structural_noise*edge_index.shape[1])
        # print("Added",num_edges_to_add," edges")
        # edge_index_to_add = torch.randint(0, len(features), (2, num_edges_to_add), dtype=torch.long)
        # edge_index_to_add = to_undirected(edge_index_to_add)
        # edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)

        edge_index = dropout_adj(edge_index,p=args.add_structural_noise,force_undirected=True)[0]

    node_names_list.append(node_names)

    #create pyg Data Object
    data = Data(x=features, edge_index = edge_index, y = y, node_names = node_names)
    #data.train_idx = idx_train
    #data.test_idx = idx_test
    #data.val_idx = idx_val 
    data_list.append(data)

meta_x = meta_x[:len(node2idx)]
number_nodes_each_graph.append(meta_x.shape[0])

# Ensure that node_names_list is not empty before concatenating
if len(node_names_list) > 0:
    node_names = np.concatenate(node_names_list, axis=0)
else:
    raise ValueError("node_names_list is empty. No arrays to concatenate.")

# Initialize meta_node_names
meta_node_names = np.zeros(len(node2idx), dtype="object")

# Populate meta_node_names based on node2idx
for k, v in node2idx.items():
    print("k:", k)  # Debugging: Print the contents of k
    if isinstance(k, tuple) and len(k) >= 2:
        k_combined = "".join(k)
        meta_node_names[v] = k_combined
    else:
# Handle case where k might not be a tuple, or has fewer elements
        meta_node_names[v] = str(k)
    
    #k1, k2 = k
    #meta_node_names[v] = k2

meta_y = torch.tensor(meta_y[:len(node2idx)]).type(torch.LongTensor)   #keep meta nodes
y = torch.concat(y_list,dim=0)
y = torch.concat([y,meta_y],dim=0)

train_set = {tuple(node) for t in train_nodes for node in t} # we convert to set to remove duplicate notes across graphs
t2 = {tuple(node) for t in val_nodes for node in t}
t3 = {tuple(node) for t in test_nodes[:-1] for node in t}
train_set = train_set.union(t2)
train_set = train_set.union(t3)

#add 10% of the training set to the val set 
val_set = set()
for i, val in enumerate(itertools.islice(train_set, int(len(train_set)*0.1))):
    val_set.add(val)
#val_set = {tuple(node) for t in val_nodes for node in t}

#test_set = {tuple(node) for t in test_nodes for node in t}
test_set = {tuple(node) for node in test_nodes[-1]} #test only in nodes from last graph

train_set = train_set - train_set.intersection(test_set) #remove test nodes from training set
val_set = val_set - val_set.intersection(test_set) #remove test nodes from val set
train_set = train_set - val_set.intersection(val_set) #remove val nodes from train set    


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
# print(batch.num_graphs)
# print(batch.x.shape)
# print(batch.y.shape)
# print(batch.batch)
# print(len(node2idx))

# print(batch.edge_index.shape)

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
              node2idx=node2idx,
              edge_indices=edge_indices_list)


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

with open(f"{model_dir}/edge_indices.pkl", "wb") as handle:
    pickle.dump([edge_index.cpu() for edge_index in edge_indices_list], handle)

final_edge_index = torch.concat([batch.edge_index.cuda(),model.meta_edge_index.cuda()],dim=1).cuda()
with open(f"{model_dir}/final_edge_index.pkl","wb") as handle:
    pickle.dump(final_edge_index.cpu(),handle)
