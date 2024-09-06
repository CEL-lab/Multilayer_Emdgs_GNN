import h5py
import os
import numpy as np
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from datetime import datetime
from pathlib import Path

def load_hdf_data(path, feature_name='node_features'):
    """Load a GCN input HDF5 container and return its content.

    This function reads an already preprocessed dataset containing all the
    data needed for training a GCN model. It extracts multiple network layers,
    features for all nodes, the names of the nodes (genes), and training,
    testing, and validation splits.

    Parameters:
    ---------
    path:               Path to the container
    feature_name:       The name of the features of the nodes. Default is: 'features'

    Returns:
    A tuple with all of the data in the order: edge_indices (list of network layers), features, y_train, y_val,
    y_test, train_mask, val_mask, test_mask, node names, feature names.
    """
    with h5py.File(path, 'r') as f:
        # Load multiple network layers
        edge_indices = []
        for i in range(1, 7):  # Assuming 6 network layers
            #print(f"Loading network_layer{i}...")
            edge_index = f[f'network_layer{i}'][:]
            #print(f"network_layer{i} shape: {edge_index.shape}")
            edge_indices.append(edge_index)
        
        # Load features and other data
       # print("Loading features...")
        features = f['node_features'][:]
        #print("Features shape:", features.shape)
        
        #print("Loading gene names...")
        node_names = f['node_names'][:]
        #print("node_names contents:", node_names)
        #print("node_names dtype:", node_names.dtype)
        
        #print("Loading training labels...")
        y_train = f['y_train'][:]
        #print("y_train shape:", y_train.shape)
        
        #print("Loading test labels...")
        y_test = f['y_test'][:]
        #print("y_test shape:", y_test.shape)
        
        if 'y_val' in f:
            #print("Loading validation labels...")
            y_val = f['y_val'][:]
            #print("y_val shape:", y_val.shape)
        else:
            y_val = None
            #print("Validation labels not found.")
        
        #print("Loading training mask...")
        train_mask = f['train_mask'][:]
        #print("train_mask shape:", train_mask.shape)
        
        #print("Loading test mask...")
        test_mask = f['test_mask'][:]
        #print("test_mask shape:", test_mask.shape)
        
        if 'mask_val' in f:
            print("Loading validation mask...")
            val_mask = f['mask_val'][:]
            print("val_mask shape:", val_mask.shape)
        else:
            val_mask = None
            print("Validation mask not found.")
        
        if 'feature_names' in f:
            print("Loading feature names...")
            feature_names = f['feature_names'][:]
            print("feature_names shape:", feature_names.shape)
        else:
            feature_names = None
            print("Feature names not found.")
    
    # Convert node names if they are byte strings
    node_names = [i.decode("utf-8") for i in node_names]
    
    return edge_indices, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names

def write_hyper_params2(args, file_name):
    """Write hyper parameters to disk.

    Writes a set of hyper parameters of the model to disk.
    See `load_hyper_params` for information on how to load
    the hyper parameters.

    Parameters:
    ----------
    args:               The parameters to save as dictionary
    input_file:         The input data hdf5 container. Only
                        present for legacy reasons
    file_name:          The file name to write the data to.
                        Should be 'hyper_params.txt' in order
                        for the load function to work properly.
    """
    with open(file_name, 'w') as f:
        for arg in vars(args):
            f.write('{}\t{}\n'.format(arg, getattr(args, arg)))
    print("Hyper-Parameters saved to {}".format(file_name))

def write_hyper_params(args, input_file, file_name):
    """Write hyper parameters to disk.

    Writes a set of hyper parameters of the model to disk.
    See `load_hyper_params` for information on how to load
    the hyper parameters.

    Parameters:
    ----------
    args:               The parameters to save as dictionary
    input_file:         The input data hdf5 container. Only
                        present for legacy reasons
    file_name:          The file name to write the data to.
                        Should be 'hyper_params.txt' in order
                        for the load function to work properly.
    """
    with open(file_name, 'w') as f:
        for arg in args:
            f.write('{}\t{}\n'.format(arg, args[arg]))
        f.write('{}\n'.format(input_file))
    print("Hyper-Parameters saved to {}".format(file_name))


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def load_hyper_params(model_dir):
    """Loads a set of hyper parameters from disk.

    Loads a set of hyper params from disk that were stored before
    by the 'write_hyper_params' function.

    Parameters:
    ----------
    model_dir:          The directory in which the model and the hyper
                        parameter text file are located.
    Returns:
    A dictionary containing the hyper parameters.
    """
    file_name = os.path.join(model_dir, 'hyper_params.txt')
    input_file = None
    with open(file_name, 'r') as f:
        args = {}
        for line in f.readlines():
            if '\t' in line:
                key, value = line.split('\t')
                if value.startswith('['): # list of hidden dimensions
                    f = lambda x: "".join(c for c in x if c not in ['\"', '\'', ' ', '\n', '[', ']']) 
                    l = [int(f(i)) for i in value.split(',')]
                    args[key.strip()] = l
                else:
                    args[key.strip()] = str_to_num(value.strip())
            else:
                input_file = line.strip()
    #print ("Hyper-Parameters read from {}".format(file_name))
    return args, input_file



def create_model_dir(model=None):
    root_dir = './results/my_models/'
    if not os.path.isdir(root_dir):  # in case training root doesn't exist
        os.mkdir(root_dir)
        print("Created Training Subdir")
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if(model is not None):
        date_string = f'{model}_{date_string}'
    os.mkdir(os.path.join(root_dir, date_string))
    print("Created dir",os.path.join(root_dir, date_string))
    model_path = os.path.join(root_dir, date_string)
    return model_path

def save_predictions(output_dir, node_names, predictions):
    p = Path(output_dir)
    p.mkdir(parents=True,exist_ok=True)
    with open(os.path.join(output_dir, 'predictions.tsv'), 'w') as f:
        f.write('ID\tName\tProb_pos\n')
        for pred_idx in range(predictions.shape[0]):
            name = node_names[pred_idx]
            f.write('{}\t{}\t{}\n'.format(name, name, predictions[pred_idx, 1]))

def write_train_test_sets(out_dir, y_train, y_test, train_mask, test_mask):
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(out_dir, 'train_mask.npy'), train_mask)
    np.save(os.path.join(out_dir, 'test_mask.npy'), test_mask)


def read_train_test_sets(dir):
    y_train = np.load(os.path.join(dir, 'y_train.npy'))
    y_test = np.load(os.path.join(dir, 'y_test.npy'))
    train_mask = np.load(os.path.join(dir, 'train_mask.npy'))
    test_mask = np.load(os.path.join(dir, 'test_mask.npy'))
    return y_train, y_test, train_mask, test_mask
