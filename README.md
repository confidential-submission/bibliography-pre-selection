
# Fast Bibliography Pre-selection Based on Dual Vector Semantic Modelling

Welcome to the repository for **Fast Bibliography Pre-selection Based on Dual Vector Semantic Modelling**. This project presents a novel approach to streamline the process of bibliography compilation in academic writing by leveraging dual vector semantic modeling to improve both efficiency and accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Introduction

Bibliography compilation is an essential yet time-consuming task in academic writing. This project introduces a novel algorithm that enhances the efficiency of bibliography pre-selection by embedding bibliographic entries and user queries into the same vector space using advanced machine learning techniques. Our dual vector model captures the asymmetrical nature of citation relationships, resulting in improved precision and reduced time required to generate comprehensive and manageable subsets of references.

## Features

- **Dual Vector Embedding**: Utilizes two distinct vector models to separately encode bibliographic entries and user queries, capturing the unique characteristics of each and improving the accuracy of pre-selection.
- **High Efficiency**: Significantly reduces the time needed to compile a bibliography, especially when handling large datasets.
- **Semantic Similarity Search**: Selects relevant references based on semantic similarity rather than simple keyword matching, providing more relevant results.
- **Customizable Pre-selection**: Allows for easy adjustments to optimize the recall-size trade-off based on user needs.

## Installation

To install and run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/confidential-submission/fast-bib-preselection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fast-bib-preselection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Pre-selecting Bibliography

To use the bibliography pre-selection tool, provide your query in the form of a title, abstract, or keywords:

```python
from fastbib import BibliographyPreSelector

preselector = BibliographyPreSelector()
subset = preselector.preselect_bibliography(query="Your research title or abstract here")
```

### Customizing Parameters

You can customize the parameters for pre-selection, such as the maximum subset size or the similarity threshold:

```python
subset = preselector.preselect_bibliography(query="Your research title", max_subset_size=50000, similarity_threshold=0.8)
```

## Training

To train the dual vector model on your own dataset, use the `train_fastbib.py` script. This script is designed to handle the training process, including loading data, defining the loss function, and updating the model's weights.

### Example Training Script Usage

1. Prepare your dataset as a list of dictionaries, where each dictionary contains a 'query', 'entry', and 'label' (1 for a positive citation link, 0 for a negative).

2. Run the training script:

   ```bash
   python train_fastbib.py
   ```

   The script will train the model and print the average loss per epoch.

### Customizing Training Parameters

You can customize various parameters in the training script, such as batch size, learning rate, and the number of epochs.

## Methodology

### Problem Definition

The problem of bibliography pre-selection is defined as finding a subset of a large bibliographic database that includes all necessary references for a given research topic while minimizing unnecessary entries. The dual vector semantic model is proposed to address this, separating the encoding processes for bibliographic entries and user queries to capture the asymmetry in citation relationships.

### Dual Vector Embedding

The core innovation of this project is the dual vector embedding technique, which uses two distinct models to encode the citing and cited works. This allows the model to better understand the context and relevance of citations, leading to more accurate pre-selection results.

### Training and Model Architecture

The model is trained on a large dataset of 3.37 million bibliography entries, using a Siamese network architecture with asymmetric feature extractors. Negative sampling is employed during training to handle the imbalance between cited and uncited entries.

### Fast Vector Search

A K-D Tree is used for fast vector search, enabling efficient selection of the most relevant bibliography entries based on the user's query.

## Experiments

### Research Questions

Our experiments focus on the following key research questions:

1. What is the minimum subset size required to achieve full recall?
2. What is the recall-size trade-off, and how does it function?
3. How does the dual vector model perform with different machine learning techniques?

### Dataset

Experiments were conducted on a subset of the Semantic Scholar database, focusing on ACL Anthology entries. Additional tests were performed using the PubMed Diabetes dataset to evaluate cross-domain applicability.

## Results

The dual vector model demonstrates superior performance in terms of recall and efficiency compared to traditional keyword-based methods and single-vector models. The experiments show that the dual vector approach achieves a better recall-size trade-off, making it an effective tool for bibliography pre-selection.

## Citation

If you use this work in your research, please cite the following paper:

```
@inproceedings{anonymous2024fast,
  title={Fast Bibliography Pre-selection Based on Dual Vector Semantic Modelling},
  author={Anonymous Author(s)},
  year={2024},
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
