# Character-Level Language Modeling with Neural Networks

This repository contains three Jupyter Notebooks demonstrating different approaches to character-level language modeling using neural networks.  Each notebook explores a distinct method, ranging from simple bigram models to more complex Multilayer Perceptrons (MLPs).

## Project Structure

* `sentence_generation.ipynb`:  Word-level language model using a neural network trained on the Brown corpus (science fiction category).
* `MLP.ipynb`: Character-level language model using a Multilayer Perceptron (MLP) trained on a dataset of names.
* `bigram_using_counts_and_NNs.ipynb`: Bigram language modeling using both frequency counts and a neural network trained on a dataset of names.
* `names.txt`:  Dataset of names used in `MLP.ipynb` and `bigram_using_counts_and_NNs.ipynb`.


## Notebooks Description

### `sentence_generation.ipynb`

This notebook builds a word-level language model using an MLP trained on sentences from the "science_fiction" category of the Brown corpus from NLTK.  Key features:

* **Data Preparation:** Loads the Brown corpus, preprocesses the text (lowercasing, alphabetic character filtering), creates word-to-index and index-to-word mappings, and builds training, development, and test datasets.  Uses a fixed block size of 8 words as input.
* **Model Architecture:** Implements a neural network with embedding, hidden, and output layers.  Uses matrix multiplication and a tanh activation function.
* **Training:** Employs mini-batch gradient descent to optimize the network parameters.
* **Evaluation:** Generates random sentences based on the learned probabilities and visualizes the training loss curve.
* **Libraries:** `torch`, `random`, `nltk`, `matplotlib`

### `MLP.ipynb`

This notebook demonstrates a character-level language model using a Multilayer Perceptron (MLP) trained on a dataset of names. Key features:

* **Data Preparation:** Reads names from `names.txt`, shuffles them, and splits them into training, development, and test sets. Converts names into numerical sequences using character-to-index mapping. Uses a block size of 3 consecutive characters as input.
* **Model Architecture:** Defines an MLP with embedding, hidden, and output layers.
* **Training:** Utilizes mini-batch gradient descent with a learning rate schedule and cross-entropy loss.
* **Evaluation:** Visualizes the training loss and the trained embedding matrix. Generates example names based on the trained model.
* **Libraries:** `torch`, `torch.nn.functional`, `matplotlib`, `seaborn`

### `bigram_using_counts_and_NNs.ipynb`

This notebook explores bigram language modeling using both frequentist and neural network approaches, trained on a dataset of names.  Key features:

* **Frequentist Approach:** Calculates bigram frequencies from the names, creates a probability matrix, and generates sequences using `torch.multinomial`.  Calculates log-likelihood and negative log-likelihood (NLL).
* **Neural Network Approach:** Creates a training dataset of bigrams using one-hot encoding. Trains a weight matrix using mini-batch gradient descent with a regularization term, minimizing NLL loss.
* **Evaluation:** Generates new name samples using both the frequentist and neural network models, and visualizes the bigram counts using a heatmap.
* **Libraries:** `torch`, `seaborn`, `matplotlib`


## Dependencies

The notebooks rely on the following Python libraries:

* `torch`
* `nltk`
* `matplotlib`
* `seaborn`
* `random`


You can install them using pip:

```bash
pip install torch nltk matplotlib seaborn
```

For NLTK, you might need to download the Brown corpus:

```python
import nltk
nltk.download('brown')
```

## Usage

1. Clone this repository.
2. Install the necessary libraries.
3. Open the Jupyter Notebooks and run the code cells sequentially.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---
Generated with ❤️ using [GitDocs](https://github.com/mikhail-ram/gitdocs).
