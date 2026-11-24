
# RAGGuard
### A Transformer-Based Framework for Detecting Hallucinations in Retrieval-Augmented Generation Systems

RAGGuard is a machine learning project designed to identify hallucinations in AI-generated responses by evaluating their consistency with a provided source context.

The project explores multiple machine learning and deep learning approaches, progressing from traditional text-based classification methods to a Transformer-based model using ModernBERT.

## Project Objective

Retrieval-Augmented Generation (RAG) systems generate responses using external information sources. However, generated responses may contain unsupported or contradictory information.

RAGGuard aims to detect such hallucinations by analyzing two inputs:

- **Source Context:** The retrieved passage provided to the AI system.
- **Generated Response:** The response produced by the AI system.

The system classifies the response into:

- **Supported (0):** No hallucination annotation detected.
- **Hallucinated (1):** The response contains a hallucination annotation.

## Methodology

The project follows an experimental approach involving multiple model architectures.

### 1. Dataset Exploration
- Dataset: RAGTruth
- Response records: 17,790
- Source passages: 2,965
- Analysis of hallucination annotations, response quality, duplicates, and class distribution.

### 2. Data Preprocessing
- Source-response pairing using source IDs.
- Removal of unsuitable records based on quality labels.
- Whitespace normalization.
- Binary label creation.
- Training, validation, and testing splits.

### 3. Machine Learning Models
Traditional machine learning approaches were evaluated using TF-IDF features.

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

### 4. Deep Learning Experiments
Additional experiments explored neural network architectures:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)

### 5. Transformer-Based Approach
ModernBERT-base was fine-tuned for binary classification using paired source and response inputs.

The model predicts whether the generated response is supported by its provided source context.

## ModernBERT Results

Performance on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 81.12% |
| Hallucination Precision | 74.07% |
| Hallucination Recall | 71.47% |
| Hallucination F1-score | 72.75% |

These results represent the current ModernBERT experiment. Additional comparisons and analysis are documented in the project notebooks and Results folder.

## Streamlit Application

RAGGuard includes a Streamlit interface where users can:

1. Paste a source passage.
2. Paste an AI-generated response.
3. Analyze the response.
4. View the predicted class.
5. View prediction confidence and class probabilities.

The application evaluates consistency with the supplied source. It does not independently verify factual truth.

## Project Structure

```text
RAGGuard/
│
├── Dataset/
│   ├── response.jsonl
│   └── source_info.jsonl
│
├── Models/
│   ├── modernbert/
│   └── modernbert_final/
│
├── Notebooks/
│   ├── 01_RAGTruth_EDA.ipynb
│   ├── 02_RAGTruthPreprocessing.ipynb
│   ├── 03_ANN.ipynb
│   ├── 04_Transformer_DeBERTa.ipynb
│   └── 06_ModernBERT.ipynb
│
├── Processed data/
│
├── Results/
│
├── src/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   └── predictor.py
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashikarathi24/RagGuard.git
cd RagGuard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Make sure the trained ModernBERT model is available at the model path configured in the application.

## Experimental Approach

RAGGuard retains the notebooks and experimental results from different stages of development. This allows the progression from exploratory data analysis and traditional machine learning to neural networks and Transformer-based classification to be examined.

## Limitations

- The classifier depends on the availability of source context.
- Predictions indicate consistency with the supplied source, not independent factual verification.
- The model performs response-level binary classification.
- Fine-grained hallucination span localization is not currently implemented.
- Prediction confidence is a model probability and is not necessarily a calibrated measure of correctness.

## Future Scope

- Fine-grained hallucination span detection.
- Claim-level verification.
- Improved handling of long source passages.
- Model explainability.
- Evaluation on additional hallucination detection datasets.

## Dataset and References

**RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models**

Niu et al., ACL 2024.

https://aclanthology.org/2024.acl-long.585/

## License

To be added.
```

This is our initial README. We'll refine it after the repository organization is complete. In particular, we'll verify the notebook filenames and the final model path before treating the structure as definitive.

## 3. Check that both files exist

Run:

```powershell
Get-ChildItem README.md, requirements.txt
```

Then:

```powershell
git status --short
```