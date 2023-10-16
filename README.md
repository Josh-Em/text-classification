# BERT Text Classification

This repository demonstrates the process of training and evaluating a BERT model for text classification tasks using the 20 Newsgroups dataset and the IMDb movie review dataset. With this code, BERT is fine-tuned on these datasets to classify news articles into one of the 20 newsgroups categories and movie reviews as positive or negative. Additionally, a trained BERT model can be applied to classify new, unseen data.

## Contents

- Preparation and tokenization of the datasets
- Creating DataLoader for training and validation sets
- Loading the pre-trained BERT model for sequence classification
- Training and evaluating the BERT model
- Measuring model performance using accuracy and classification report
- Classifying a new text input using the trained BERT model

**Note**: There are separate script files for each classification task (multi-class and binary). Please refer to the appropriate script based on your use case.

## Getting Started

This project makes use of [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers) libraries. To run the code, please ensure you have these libraries installed in your environment.

### Prerequisites

- Python 3.6 or later
- PyTorch
- Transformers
- NumPy
- pandas
- scikit-learn
- tqdm
- Datasets

### Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/) following the instructions for your system and desired configuration.

2. Install the other required packages:

```bash
pip install transformers numpy pandas scikit-learn tqdm datasets
```

## Code Examples

1. Clone this repository and open the respective Python script or Colab Notebook based on the desired classification task (20 Newsgroups or IMDb dataset). Alternatively, you can run the given code in any Python environment with the required libraries.

2. Follow the code in the script or notebook to train and evaluate the BERT model for the selected text classification task.

3. Test the model with a new text input by modifying the `sample_text` string in the provided code block.

4. Evaluate the model's performance using the classification report and accuracy metric to get insights on how well the model generalizes to unseen data.

## Next Steps

- Consider training the BERT model for other text classification tasks by modifying the dataset and adapting the code accordingly.
- Experiment with different variations of the BERT architecture (e.g., `bert-large-uncased`) or different pre-trained transformer models for additional improvements.
- Optimize the model's hyperparameters (e.g., learning rate, batch size) to enhance its performance.
