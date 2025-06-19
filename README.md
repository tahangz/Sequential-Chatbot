# Sequential-Chatbot
## Chatbot with GRU Encoder-Decoder and Attention

A sequence-to-sequence chatbot built using PyTorch, leveraging GRU-based encoder and decoder with attention mechanism. Trained on the Cornell Movie-Dialogs Corpus (approximately 24k conversational pairs), this project demonstrates an end-to-end pipeline: data preprocessing, model training, evaluation, and interactive inference (chat).

---

### 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Interactive Chat](#interactive-chat)
9. [Evaluation](#evaluation)
10. [Results and Insights](#results-and-insights)
11. [Future Work](#future-work)
12. [Contributing](#contributing)
13. [License](#license)

---

## Project Overview

This repository contains the implementation of a chatbot using a GRU-based encoder-decoder architecture with attention. The model is trained on the Cornell Movie-Dialogs Corpus, which provides roughly 24,000 conversational question-answer pairs from movie scripts. The goal is to build a conversational agent that can generate contextually relevant responses to user inputs.

Key features:

* Data loading and preprocessing scripts for the Cornell Movie-Dialogs Corpus.
* GRU-based Encoder and Decoder with attention (Luong attention).
* Training loop with checkpointing, loss tracking, and optional GPU support.
* Interactive chat interface for real-time inference.
* Modular, readable code adhering to best practices in software engineering.

---

## Dataset

We use the Cornell Movie-Dialogs Corpus:

* Link: [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* Contains \~24,000 question-answer pairs (one-turn dialogues) for training.

Ensure you download and place the dataset files in the appropriate data directory as explained below.

---

## Requirements

* Python 3.7+
* PyTorch (1.7+ recommended)
* NumPy
* tqdm
* Pickle (comes with Python)

Optionally for GPU training:

* CUDA-enabled GPU with compatible PyTorch build.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/tahangz/Sequential-Chatbot.git
   cd Sequential-Chatbot
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download Dataset**

   * Download the Cornell Movie-Dialogs Corpus from: [https://www.cs.cornell.edu/\~cristian/Cornell\_Movie-Dialogs\_Corpus.html](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
   * Extract and place files (e.g., `movie_lines.txt`, `movie_conversations.txt`) into `data/raw/`.

---

## Data Preprocessing

1. **Filtering and Pair Extraction**

   * Use preprocessing scripts to read `movie_lines.txt` and `movie_conversations.txt`, normalize text (lowercasing, trimming punctuation), and extract question-answer pairs.
   * Apply a maximum sentence length filter (e.g., 10 tokens) to focus on concise exchanges.
   * Save filtered pairs to a processed data file (e.g., `pairs.txt`).

2. **Build Vocabulary**

   * Tokenize sentences, count word frequencies.
   * Create mappings: `word2index`, `index2word`, with special tokens (`PAD`, `SOS`, `EOS`, `UNK`).
   * Serialize the `voc` object (e.g., via pickle) for later use.

Example usage:

```bash
python data_preprocessing.py --data_dir data/raw --save_dir data/processed --max_length 10
```

---

## Model Architecture

* **Encoder (EncoderRNN)**: GRU-based RNN that processes the input sequence.

  * Input: embedded word vectors.
  * Hidden size: 500.
  * Number of layers: 2.
  * Dropout: 0.1 between layers if >1 layer.
  * Bidirectional optional (if implemented).

* **Decoder (LuongAttnDecoderRNN)**: GRU-based RNN with attention.

  * Uses Luong-style attention (`dot`, `general`, or `concat`).
  * Inputs: previous word embedding + context vector from attention.
  * Hidden size: 500.
  * Number of layers: 2.
  * Dropout: 0.1.
  * Output: distribution over vocabulary for next token.

* **Greedy Search Decoder** for inference (can be extended to beam search).

All model definitions reside in the notebook or corresponding scripts.

---

## Training

1. **Configure Hyperparameters**:

   * `hidden_size=500`
   * `encoder_n_layers=2`
   * `decoder_n_layers=2`
   * `dropout=0.1`
   * `batch_size=64`
   * `learning_rate=0.0001`
   * `n_epochs=10-20` (monitor validation loss)
   * `teacher_forcing_ratio=0.5`

2. **Run Training**

   In Colab notebook:

   ```python
   # After loading data and vocabulary
   train(..., hidden_size=500, encoder_n_layers=2, decoder_n_layers=2, dropout=0.1,
         batch_size=64, learning_rate=0.0001, n_epochs=15, teacher_forcing_ratio=0.5)
   ```

3. **Checkpointing**

   * Checkpoints saved periodically to Google Drive or Colab session storage.
   * Ensure to save:

     * Encoder state\_dict
     * Decoder state\_dict
     * Vocabulary object
     * Training iteration count

4. **GPU Support**

   * Colab provides GPU; ensure runtime is set to GPU for faster training.

---

## Interactive Chat

After training in Colab, use the `chat` function defined in the notebook to launch an interactive session within the notebook:

```python
# After loading models and vocabulary
chat(encoder, decoder, voc)
```

Inside `chat`:

1. Models are set to evaluation mode (`model.eval()`).
2. Read user input, preprocess (normalize, tokenize, convert to tensor).
3. Run encoder and decoder with attention to generate response (greedy search by default).
4. Display bot response; repeat until user exits (e.g., typing 'quit').

Example interaction:

```
User: hello
Bot: hi . how are you ?
User: i am fine
Bot: that 's good to hear .
User: quit
Bot: Goodbye!
```

---

## Evaluation

* **Qualitative**: Interact with the chatbot in Colab to inspect response quality.
* **Quantitative**: (Optional) Compute metrics like BLEU on a validation set.
* **Loss Curves**: Plot training/validation loss within the notebook to observe convergence.

---

## Results and Insights

* After training \~15 epochs on \~24k pairs, expect basic conversational ability but potentially generic responses.
* Attention mechanism improves relevance by focusing on input tokens.
* Consider larger datasets or advanced architectures for diversity.

---

## Future Work

* **Beam Search**: Implement beam search decoding for potentially better responses.
* **Transformer Models**: Explore transformer-based seq2seq for improved performance.
* **Contextual History**: Extend to multi-turn dialogues by maintaining conversation context across turns.
* **Pretrained Embeddings**: Integrate GloVe or contextual embeddings (e.g., BERT) to improve generalization.
* **Deployment**: Export models and deploy via API (e.g., Flask/FastAPI) for integration into applications.
* **Safety & Filtering**: Add mechanisms to filter inappropriate outputs.

---

