import jsonlines
from transformers import AutoTokenizer, pipeline
from FlagEmbedding import BGEM3FlagModel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import kaldi_io
import numpy as np
import itertools
import torch


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
# This script builds three parallel Kaldi-style ark/scp archives for a given
# split of the MLSUM dataset:
#   1) sentence embeddings of the source document (BGE-M3)
#   2) tokenized + padded target summary (BARThez tokenizer)
#   3) tokenized named-entity "prefix" extracted from the source document
# Each entry's records across the three archives share the same key
# (entry_id), so they can be re-joined later at training time.
mode = "TRAIN"

# Input JSONL for this split (keys expected: id, text, summary).
input_file = f"/path_to_folder/mlsum_{mode}.jsonl"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
# All generated ark/scp files are written into <repo_root>/dataset/data/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
output_dir = os.path.join(PROJECT_ROOT, "dataset", "data")
os.makedirs(output_dir, exist_ok=True)

embedding_ark_file = os.path.join(output_dir, f"mlsum_{mode}_embeddings.ark")
embedding_scp_file = os.path.join(output_dir, f"mlsum_{mode}_embeddings.scp")
tokens_ark_file = os.path.join(output_dir, f"mlsum_{mode}_tokens.ark")
tokens_scp_file = os.path.join(output_dir, f"mlsum_{mode}_tokens.scp")
ner_ark_file = os.path.join(output_dir, f"mlsum_{mode}_ner.ark")
ner_scp_file = os.path.join(output_dir, f"mlsum_{mode}_ner.scp")


# ---------------------------------------------------------------------------
# Initialize models
# ---------------------------------------------------------------------------
# BARThez tokenizer is used for the *targets* (summary + NER prefix), since
# the downstream model this data feeds into is a BARThez-based seq2seq summarizer.
target_tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
# BGE-M3 embedding model, used to embed each sentence of the source text.
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Initialize NER pipeline using the CamemBERT-based French NER system.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ner_pipeline = pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple", device=device)


def extract_named_entities_hf(text, threshold=0.9):
    """
    Extracts named entities with types and confidence >= threshold.
    Removes redundancy based on entity name only, not type.
    Returns a list of (entity, type) tuples.
    """
    ner_results = ner_pipeline(text)
    seen_names = set()
    named_entities = []

    for ent in ner_results:
        if ent['score'] >= threshold:
            # Title-case normalization so that e.g. "paris" and "Paris"
            # collapse to the same seen-name key.
            entity = ent['word'].strip().title()
            if entity not in seen_names:
                seen_names.add(entity)
                named_entities.append((entity, ent['entity_group']))

    return named_entities if named_entities else [("Inconnu", "UNK")]


def generate_dynamic_prefix(dialogue_text):
    """
    Returns named entities with types in the format:
    (E1, T1), (E2, T2), (E3, T3).
    """
    entities = extract_named_entities_hf(dialogue_text)
    return ', '.join(f"({ent}, {typ})" for ent, typ in entities) + '.'


def embed_text(sentences, model):
    """Generate embeddings for a list of sentences."""
    try:
        # max_length=4096 is generous headroom; individual split sentences
        # are expected to be well under this.
        outputs_embedding = model.encode(sentences, max_length=4096)['dense_vecs']
        return outputs_embedding.astype(np.float32)
    except Exception as e:
        # Swallow errors so one bad sentence list doesn't kill the whole run;
        # caller receives an empty array as a sentinel for "embedding failed".
        print(f"Error in embedding: {e}", flush=True)
        return np.array([])

def pad_encode(tokens, pad_token_id, max_length=512):
    """
    Pad tokens to max_length.
    """
    # Truncate first (in case caller passes something longer than max_length),
    # then right-pad with pad_token_id up to a fixed width so every record
    # written to the ark file has the same shape.
    tokens = tokens[:max_length] + [pad_token_id] * (max_length - len(tokens))
    return tokens


pad_token_id = target_tokenizer.pad_token_id if target_tokenizer.pad_token_id is not None else target_tokenizer.eos_token_id

# Locks and counters
counter = 0
counter_lock = threading.Lock()
entry_id_counter = itertools.count()

# Open all three ark files (binary) up front; each is written to
# incrementally as entries are processed.
with kaldi_io.open_or_fd(embedding_ark_file, 'wb') as emb_ark, \
     kaldi_io.open_or_fd(tokens_ark_file, 'wb') as tok_ark, \
     kaldi_io.open_or_fd(ner_ark_file, 'wb') as ner_ark:

    # Corresponding scp index files: map each entry_id -> byte offset in
    # its ark file, so records can be looked up individually later.
    with open(embedding_scp_file, 'w', encoding="utf-8") as emb_scp, \
         open(tokens_scp_file, 'w', encoding="utf-8") as tok_scp, \
         open(ner_scp_file, 'w', encoding="utf-8") as ner_scp:

        writer_lock = threading.Lock()

        def process_and_write_entry(entry):
            global counter
            transcription = entry["text"]
            reference_synopsis = entry["summary"]
            entry_id = entry["id"]
            # Uncomment the following line if the data has no key "id", this id helps to save the embeddings and tokenized targets into the same key
            # entry_id = f"entry_{next(entry_id_counter)}"

            # Generate embeddings
            # Naive sentence split on ".": good enough for MLSUM-style text,
            # but will mis-split on abbreviations/decimals if present.
            sentences = [s.strip() for s in transcription.split(".") if s.strip()]
            transcription_embeddings = embed_text(sentences, embedding_model)

            # Generate target tokens
            target_tokens = target_tokenizer.encode(reference_synopsis, truncation=True, max_length=512)
            target_tokens_padded = pad_encode(target_tokens, pad_token_id)
            target_tokens_padded = np.array(target_tokens_padded, dtype=np.float32)

            # Extract Named Entities
            ner_words = generate_dynamic_prefix(transcription)
            if ner_words.lower().startswith("(inconnu"):
                ner_tokens = np.array([-1], dtype=np.float32)
            else:
                ner_tokens = target_tokenizer.encode(ner_words, truncation=True, max_length=512)
                ner_tokens = np.array(ner_tokens, dtype=np.float32)

            # Write all three records for this entry under the same key.
            with writer_lock:
                pos = emb_ark.tell()
                kaldi_io.write_mat(emb_ark, transcription_embeddings)
                emb_scp.write(f"{entry_id} {embedding_ark_file}:{pos}\n")

                pos = tok_ark.tell()
                kaldi_io.write_vec_flt(tok_ark, target_tokens_padded)
                tok_scp.write(f"{entry_id} {tokens_ark_file}:{pos}\n")

                pos = ner_ark.tell()
                kaldi_io.write_vec_flt(ner_ark, ner_tokens)
                ner_scp.write(f"{entry_id} {ner_ark_file}:{pos}\n")

            # Lightweight progress logging every 100 entries.
            with counter_lock:
                counter += 1
                if counter % 100 == 0:
                    print(f'Processed {counter} entries', flush=True)

        # Process in parallel
        with jsonlines.open(input_file, "r") as reader:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_and_write_entry, entry) for entry in reader]
                for future in as_completed(futures):
                    future.result()

print(f"Processing complete. Files saved.", flush=True)