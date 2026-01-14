# Week 5 NLP Course

A comprehensive Natural Language Processing course covering statistical NLP, deep learning approaches, and advanced topics.

[📊 Click to view slides](https://drive.google.com/drive/folders/1FJT0I5skf0P8UPB5CX4TSwEHYKCb9CcG?usp=drive_link)

---

## 📚 Course Overview

This course is structured into three modules, progressing from traditional statistical NLP methods to modern deep learning approaches:

- **Module 1**: Text Analysis with Statistical NLP
- **Module 2**: NLP with PyTorch (Deep Learning)
- **Module 3**: Advanced Topics (Natural Language Inference)

---

## 🛠️ Setup

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd W5_NLP
```

2. Install dependencies using `uv sync`:
```bash
uv sync
```

This will:
- Create a virtual environment (if it doesn't exist)
- Install all project dependencies from `pyproject.toml`
- Install development dependencies (Jupyter, etc.)

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

4. Start Jupyter to run the notebooks:
```bash
jupyter notebook
```

### Key Dependencies

- **NLP Libraries**: `nltk`, `camel-tools`, `farasapy`, `pyarabic`, `qalsadi`
- **ML/Data Science**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Deep Learning**: `transformers`, `torch` (for Module 2)
- **Utilities**: `joblib`, `regex`, `tiktoken`

---

## 📁 Project Structure

```
W5_NLP/
├── M1/                          # Module 1: Statistical NLP
│   └── sessions/                # Main course sessions
│       ├── 05_classifier_ex1.ipynb
│       ├── 06_classifier_ex2.ipynb
│       ├── 07_information_retrieval_ex.ipynb
│       └── 08_topic_modeling_ex.ipynb
│
├── M2/                          # Module 2: Deep Learning NLP
│   ├── lessons/                 # Lesson notebooks
│   │   └── 01_M2_intro.ipynb
│   └── labs/                    # Hands-on labs
│       ├── C2_M3_Lab_1_basic_tokenization/
│       ├── C2_M3_Lab_2_embeddings/
│       └── C2_M3_Lab_4_finetuned_text_classifier/
│
├── M3/                          # Module 3: Advanced Topics
│   └── labs/
│       └── 02_NLI.ipynb
│
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

---

## 📖 Learning Path

### For Beginners
1. Start with **Module 1, Session 1** (Introduction to NLP)
2. Work through sessions sequentially
3. Complete all exercises before moving to Module 2

### For Intermediate Learners
1. Review Module 1 sessions 1-4 (foundations)
2. Focus on sessions 5-8 (applications)
3. Proceed to Module 2 for deep learning approaches

### For Advanced Learners
1. Quick review of Module 1 key concepts
2. Deep dive into Module 2 labs
3. Explore Module 3 advanced topics

---

## 📦 Module 1: Text Analysis with Statistical NLP

**Module Overview**: Learn how to work with text data using traditional statistical methods—the foundation that modern deep learning approaches build upon.

### Key Skills

- Build, evaluate, and improve supervised ML text classification pipelines
- Apply keyword-based information retrieval algorithms to find specific data in unstructured text
- Apply unsupervised ML to cluster documents and discover hidden topics in unorganized, unlabeled data

### Sessions

The module follows a logical progression where each session builds on the previous one:

1. **Session 1** - Introduction to NLP
   - What is NLP?
   - Statistical NLP vs. Deep Learning approaches
   - NLP pipeline overview
   - Python NLP ecosystem

2. **Session 2** - Corpora & Exploratory Data Analysis
   - Understanding your data
   - Corpus analysis techniques

3. **Session 3** - Text Preprocessing
   - Cleaning and normalizing text
   - Tokenization
   - Text preparation for ML

4. **Session 4** - Vectorization
   - Converting text to numbers
   - Bag of Words (BoW)
   - TF-IDF vectorization

5. **05_classifier_ex1.ipynb** - Text Classification Exercise 1
   - Building text classifiers
   - Evaluation metrics

6. **06_classifier_ex2.ipynb** - Text Classification Exercise 2
   - Advanced classification techniques
   - Model improvement strategies

7. **07_information_retrieval_ex.ipynb** - Information Retrieval
   - Keyword-based search algorithms
   - Document similarity and retrieval
   - Building search engines with TF-IDF

8. **08_topic_modeling_ex.ipynb** - Topic Modeling
   - Latent Dirichlet Allocation (LDA)
   - Discovering hidden topics in unlabeled documents
   - Organizing documents by topic

---

## 🧠 Module 2: NLP with PyTorch (Deep Learning)

**Module Overview**: Transition from statistical NLP to modern deep learning approaches using PyTorch and transformer models.

### Key Skills

- Understand tokenization by comparing manual approaches with pre-trained tools
- Understand embeddings by visualizing pre-trained models and building from scratch
- Distinguish how `input_ids` and `attention_mask` relate to tokens and embeddings
- Prepare custom `Dataset` + `DataCollatorWithPadding` for efficient batching
- Fine-tune pre-trained contextual embedding models (like BERT) on custom datasets for higher classification accuracy

### Lessons

- **01_M2_intro.ipynb** - From Statistical to Neural NLP
  - Tokenization evolution: word-level to subword-level
  - Vectorization evolution: sparse (BoW/TF-IDF) to dense (embeddings)
  - Static vs. contextual embeddings

### Labs

1. **C2_M3_Lab_1_basic_tokenization** - Basic Tokenization
   - Manual tokenization vs. pre-trained tokenizers
   - BERT tokenizer implementation
   - Subword tokenization (BPE, WordPiece)

2. **C2_M3_Lab_2_embeddings** - Embeddings
   - Understanding word embeddings
   - Visualizing pre-trained models
   - Building embeddings from scratch
   - Static vs. contextual embeddings

3. **C2_M3_Lab_4_finetuned_text_classifier** - Fine-tuned Text Classifier
   - Preparing custom datasets
   - Fine-tuning BERT for text classification
   - Evaluation and deployment

---

## 🚀 Module 3: Advanced Topics

**Module Overview**: Explore advanced NLP techniques including Natural Language Inference and zero-shot classification.

### Key Skills

- Use Natural Language Inference for zero-shot classification
- Apply pre-trained models for various NLP tasks without fine-tuning
- Understand entailment, contradiction, and neutral relationships

### Labs

- **02_NLI.ipynb** - Natural Language Inference
  - Zero-shot classification using NLI
  - Entailment, contradiction, and neutral classification
  - Using pre-trained NLI models from HuggingFace
  - Applications of NLI for text classification

---

## 🎯 Learning Outcomes

By completing this course, you will be able to:

1. **Understand NLP Fundamentals**
   - Differentiate between statistical and neural NLP approaches
   - Understand the complete NLP pipeline from raw text to models

2. **Apply Statistical NLP**
   - Preprocess and vectorize text data
   - Build and evaluate text classifiers
   - Implement information retrieval systems
   - Perform topic modeling on unlabeled documents

3. **Work with Deep Learning NLP**
   - Use modern tokenization techniques (subword tokenization)
   - Understand and work with embeddings (static and contextual)
   - Fine-tune pre-trained transformer models
   - Build custom datasets for NLP tasks

4. **Apply Advanced Techniques**
   - Use Natural Language Inference for zero-shot classification
   - Leverage pre-trained models for various NLP tasks

---

## 📚 References

### Module 1
- [NLP Pipeline, Ali Alameer | GitHub](https://github.com/Ali-Alameer/NLP/blob/main/week2_pipeline_part1.ipynb)
- [NLP_Getting_started(Preprocessing), Ali H. El-Kassas | Kaggle](https://www.kaggle.com/code/ali01lulu/03-nlp-getting-started-preprocessing/notebook)

### Module 2
- [PyTorch: Techniques and Ecosystem Tools](https://www.coursera.org/learn/pytorch-techniques-and-ecosystem-tools)
- [NLP Demystified](https://www.nlpdemystified.org/course)
