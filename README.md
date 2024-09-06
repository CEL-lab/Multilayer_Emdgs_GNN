# Multilayer GNNs and Network Embeddings for Predictive Analysis in Energy Networks

This repository contains the Python code and resources for the paper **"Multilayer GNNs and Network Embeddings for Predictive Analysis in Energy Networks."** The project aims to apply **Multilayer Graph Neural Networks (GNNs)** and **Network Embeddings** to incident data recorded on PMUs (Phasor Measurement Units) from OGE (Oklahoma Gas & Electric) between 2015 and 2021. The project is divided into three key parts:

1. **Classical Machine Learning**: Using raw data and selected features for predictive analysis.
2. **Multilayer Network Embeddings**: Using the **multi-node2vec** algorithm to embed multilayer network structures.
3. **Multilayer GNNs**: Applying **Explainable Multilayer GNNs (EMGNN)** for node-level and edge-level predictions.

## Data Preparation

The incident data used in this project spans from **2015 to 2021** and includes fields such as:

| Column                       | Description                                              |
|-------------------------------|----------------------------------------------------------|
| Job Display ID                | Unique identifier for each job.                          |
| CAD_ID                        | Incident ID.                                             |
| Job Region                    | Region of the job.                                       |
| Job Area (DISTRICT)           | District where the job took place.                       |
| Job Substation                | Substation ID.                                           |
| ...                           | Other columns related to incident duration, customers affected, causes, and equipment descriptions. |

### Network Construction

We constructed a **six-layer multilayer network** from the incident data, with each layer representing a specific feature of the data:

- `Job Region`
- `Month/Day/Year`
- `Custs Affected`
- `OGE Causes`
- `Major Storm Event (Yes/No)`
- `Distribution, Substation, Transmission`

These layers were preprocessed and saved into an `.h5` container that contains the following:

- Node features (numeric and one-hot encoded non-numeric features)
- Six network layers (edge indices for each layer)
- Train/test masks and target classes

## Classical Machine Learning and Network Embeddings

The entire process of classical machine learning and generating network embeddings is performed in the `Final_Preprocess_NW_Embeddings.ipynb` notebook. This notebook contains detailed explanations for each step, including:

1. **Network Construction**: Creating adjacency matrices for the multilayer network.
2. **Network Embedding**: Using the **multi-node2vec** algorithm to generate network embeddings from the multilayer structure.
3. **Classical Machine Learning**: Applying **RandomForest**, **XGBoost**, and **KNeighbors** classifiers on the **full raw data** and **selected features**, as well as on the embeddings generated from the multilayer network.

Every step, from preprocessing the data to merging the network embeddings with the target column and running classical machine learning models, is covered in the `.ipynb` file.

### How to Run

You can run the entire process, including classical machine learning and embedding generation, by executing the Jupyter notebook `Final_Preprocess_NW_Embeddings.ipynb`.

Alternatively, you can generate network embeddings separately by running:

```bash
python multi_node2vec.py --dir <path_to_network_files> --output <path_to_save_embeddings> --d 100 --window_size 10
```
For more details on multi-node2vec, you can refer to the paper [here](https://github.com/jdwilson4/multi-node2vec).

## 3. Multilayer GNN (EMGNN)
In the final part of the project, we used a Multilayer GNN (EMGNN) to capture complex relationships across multiple layers of the network. EMGNN helps us to predict target classes (Job Area District) by learning both node-level and edge-level relationships across layers.

### How to Train
To train the Multilayer GNN, run:
```bash
python train.py --gcn 1 --dataset <dataset_name>
```
Replace <dataset_name> with the name of your preprocessed multilayer network in .h5 format.

## Explainability
You can explain the model predictions using Captum by running the following command:
```bash
python explain.py --model_dir <path_to_trained_model> --gene_label <target_class>
```
This will generate insights into the model's predictions by explaining edge and node feature importance.

## Requirements
To run the project, ensure you have Python 3.x installed along with the required dependencies. You can install them by running:
```bash
pip install -r requirements.txt
```
### Key Libraries:
- Python 3.9
- PyTorch
- CuML (for GPU-accelerated machine learning)
- PyTorch Geometric
- Captum
- NetworkX
- Scikit-learn
- Pandas

## Citation

If you use this work, please cite the following:

**Muhammad Kazim, Harun Pirim**  
*Multilayer GNNs and Network Embeddings for Predictive Analysis in Energy Networks*  
*[Paper in preparation]*
