import jsonlines
from transformers import AutoTokenizer, pipeline
from FlagEmbedding import BGEM3FlagModel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import kaldi_io
import numpy as np
import itertools
import torch


# File paths
mode = "TRAIN"
input_file = f"/path_to_folder/mlsum_{mode}.jsonl"
embedding_ark_file = f"/path_to_folder/mlsum_{mode}_embeddings.ark"
embedding_scp_file = f"/path_to_folder/mlsum_{mode}_embeddings.scp"
tokens_ark_file = f"/path_to_folder/mlsum_{mode}_tokens.ark"
tokens_scp_file = f"/path_to_folder/mlsum_{mode}_tokens.scp"
ner_ark_file = f"/path_to_folder/mlsum_{mode}_ner.ark" 
ner_scp_file = f"/path_to_folder/mlsum_{mode}_ner.scp" 


# Initialize models
target_tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Initialize NER pipeline
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
        outputs_embedding = model.encode(sentences, max_length=4096)['dense_vecs']
        return outputs_embedding.astype(np.float32)
    except Exception as e:
        print(f"Error in embedding: {e}", flush=True)
        return np.array([])

def pad_encode(tokens, pad_token_id, max_length=512):
    """
    Pad tokens to max_length.
    """
    tokens = tokens[:max_length] + [pad_token_id] * (max_length - len(tokens))
    return tokens


pad_token_id = target_tokenizer.pad_token_id if target_tokenizer.pad_token_id is not None else target_tokenizer.eos_token_id

# Locks and counters
counter = 0
counter_lock = threading.Lock()
entry_id_counter = itertools.count()

with kaldi_io.open_or_fd(embedding_ark_file, 'wb') as emb_ark, \
     kaldi_io.open_or_fd(tokens_ark_file, 'wb') as tok_ark, \
     kaldi_io.open_or_fd(ner_ark_file, 'wb') as ner_ark:

    with open(embedding_scp_file, 'w', encoding="utf-8") as emb_scp, \
         open(tokens_scp_file, 'w', encoding="utf-8") as tok_scp, \
         open(ner_scp_file, 'w', encoding="utf-8") as ner_scp:

        writer_lock = threading.Lock()

        def process_and_write_entry(entry):
            global counter
            transcription = entry["text"]
            reference_synopsis = entry["summary"]
            entry_id = f"entry_{next(entry_id_counter)}"

            # Generate embeddings
            sentences = [s.strip() for s in transcription.split(".") if s.strip()]
            transcription_embeddings = embed_text(sentences, embedding_model)

            # Generate target tokens
            target_tokens = target_tokenizer.encode(reference_synopsis, truncation=True, max_length=512)
            target_tokens_padded = pad_encode(target_tokens, pad_token_id)
            target_tokens_padded = np.array(target_tokens_padded, dtype=np.float32)

            # Extract Named Entities
            ner_words = generate_dynamic_prefix(transcription)
            if "inconnu" in ner_words:
                ner_tokens = np.array([-1], dtype=np.float32)  # Ensure it's a NumPy array of the same type
            else:
                ner_tokens = target_tokenizer.encode(ner_words, truncation=True, max_length=512)
                ner_tokens = np.array(ner_tokens, dtype=np.float32)

            with writer_lock:
                kaldi_io.write_mat(emb_ark, transcription_embeddings)
                emb_scp.write(f"{entry_id} {embedding_ark_file}:{emb_ark.tell()}\n")

                kaldi_io.write_vec_flt(tok_ark, target_tokens_padded)
                tok_scp.write(f"{entry_id} {tokens_ark_file}:{tok_ark.tell()}\n")

                kaldi_io.write_vec_flt(ner_ark, ner_tokens)
                ner_scp.write(f"{entry_id} {ner_ark_file}:{ner_ark.tell()}\n")

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