# SBARThez

## Official code for the paper "Using Multimodal and Language-Agnostic Sentence Embeddings for Abstractive Summarization"

**Abstract**  
Abstractive summarization aims to generate concise summaries by creating new sentences, allowing for flexible rephrasing. However, this approach can be vulnerable to inaccuracies, particularly 'hallucinations' where the model introduces non-existent information. In this paper, we leverage the use of multimodal and multilingual sentence embeddings derived from pre-trained models such as LaBSE, SONAR, and BGE-M3, and feed them into a modified BART-based French model. A Named Entity Injection mechanism that appends tokenized named entities to the decoder input is introduced, in order to improve the factual consistency of the generated summary. Our novel framework, SBARThez, is applicable to both text and speech inputs and supports cross-lingual summarization; it shows competitive performance relative to token-level baselines, especially for low-resource languages, while generating more concise and abstract summaries. 

---

## 📦 Usage

### ✅ Installation

Install all required packages using:

```bash
pip install -r requirements.txt
```

### 🛠️ Dataset Preparation : Generation of Sentence embeddings

To generate sentence embeddings in scp/ark format for the MLSUM dataset (or any other dataset) using the BGE-M3 model, run:

```bash
python scripts/generate_scp_ark.py
```

### 🏋️ Training 

Once the correct dataset paths and training hyperparameters are specified in configs/train_config.yaml, start the training process by running:

```bash
python scripts/train.py
```

### 🔍 Inference

To run inference using the trained model's checkpoints:

```bash
python scripts/inference.py --ckpt checkpoints/sbarthez_1.pth \
    --embeddings test_emb.scp \
    --tokens test_tok.scp \
    --ner test_ner.scp \
    --beam
```


