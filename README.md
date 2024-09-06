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

## Classical Machine Learning

In the first part of the project, we applied **RandomForest**, **XGBoost**, and **KNeighbors** classifiers on the **full raw data** and **selected features**.

### How to Run

Run the classical machine learning experiments using the script:

```bash
python classical_ml.py
----
This will train and evaluate the models on full raw data and selected features, outputting accuracy and performance metrics to the `Results` folder.

## 2. Multilayer Network Embeddings

For embedding nodes in the multilayer network, we used the **multi-node2vec** algorithm, which performs random walks across the multilayer structure and learns node representations.

### How to Run

To generate node embeddings, run:

```bash
python multi_node2vec.py --dir <path_to_network_files> --output <path_to_save_embeddings> --d 100 --window_size 10

