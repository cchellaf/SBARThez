from dataset.kaldi_dataset import KaldiDataset, collate_fn
from models.sbarthez_model import SBARThez_BGE
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import evaluate
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Run SBARThez model inference.")
parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint (.pth file)")
parser.add_argument("--emb", type=str, required=True, help="Path to the embedding .scp file")
parser.add_argument("--tok", type=str, required=True, help="Path to the token .scp file")
parser.add_argument("--ner", type=str, default=None,
                     help="Path to the NER token .scp file. Only needed when the checkpoint "
                          "was trained with with_nei=True (or --with_nei is passed explicitly).")
parser.add_argument("--beam", action="store_true", help="Use beam search decoding instead of greedy")
parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search decoding")
parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Max number of tokens to generate per example")
# --with_nei / --without_nei let the user force the setting explicitly.
# If neither is passed, we auto-detect from the checkpoint (see below).
parser.add_argument("--with_nei", dest="with_nei", action="store_true", default=None,
                     help="Force-enable the NER prefix, overriding whatever the checkpoint says.")
parser.add_argument("--without_nei", dest="with_nei", action="store_false",
                     help="Force-disable the NER prefix, overriding whatever the checkpoint says.")
args = parser.parse_args()

valid_emb_path = args.emb
valid_token_path = args.tok
checkpoint_path = args.ckpt

##################### MODEL INITIALIZATION ##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBARThez_BGE().to(device)
tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

checkpoint = torch.load(checkpoint_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_with_nei = checkpoint.get("with_nei", None)
else:
    model.load_state_dict(checkpoint)
    checkpoint_with_nei = None

if args.with_nei is not None:
    with_nei = args.with_nei
    if checkpoint_with_nei is not None and checkpoint_with_nei != with_nei:
        print(f"WARNING: checkpoint was trained with with_nei={checkpoint_with_nei}, "
              f"but --with_nei/--without_nei explicitly requests with_nei={with_nei}. "
              f"Using the CLI value.", flush=True)
elif checkpoint_with_nei is not None:
    with_nei = checkpoint_with_nei
else:
    print("WARNING: could not determine with_nei from checkpoint (old-format checkpoint "
          "with no metadata). Defaulting to with_nei=False. Pass --with_nei explicitly "
          "if this checkpoint was actually trained with the NER prefix.", flush=True)
    with_nei = False

model.eval()
print(f"Running inference with NEI module: {with_nei}", flush=True)
print(f"Batch size: {args.batch_size} | Beam search: {args.beam} (beam_size={args.beam_size})", flush=True)

if with_nei and args.ner is None:
    parser.error("--ner is required because with_nei=True (either from the checkpoint or "
                 "from an explicit --with_nei flag).")

test_dataset = KaldiDataset(valid_emb_path, valid_token_path, args.ner)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

# Evaluation Metrics
rouge = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


def build_ner_prefix_batch(ner_tokens_list, with_nei):
    """
    ner_tokens_list: list of 1D LongTensors, one per example in the batch
    (as returned by collate_fn -- lengths vary per example).

    Mirrors the exact padding scheme train_sbarthez.py uses: within a
    batch, prefixes are right-padded to the length of the *longest prefix
    in that batch* (not to a fixed max_length), and the [-1] sentinel
    ("no entities found") becomes an all-pad row. Matching this scheme
    exactly is what keeps inference consistent with what the model saw
    during training.

    Returns a (batch_size, max_prefix_len) LongTensor on `device`, or
    None if with_nei is False.
    """
    if not with_nei:
        return None

    batch_size = len(ner_tokens_list)
    max_prefix_len = max(p.shape[0] for p in ner_tokens_list)
    padded_prefixes = torch.full((batch_size, max_prefix_len), pad_token_id, dtype=torch.long, device=device)

    for i, p in enumerate(ner_tokens_list):
        if p.shape[0] == 1 and p[0] == -1:
            continue  # sentinel: no entities found for this example -> leave row as all-pad
        padded_prefixes[i, :p.shape[0]] = p.to(device)

    return padded_prefixes


def generate_batch(model, embedding, attention_mask, ner_tokens_list, with_nei, beam, beam_size, max_new_tokens):
    """
    Builds the forced decoder prefix (NER prefix + <bos>, matching
    training's layout) and delegates generation to the model's own
    .generate(), batched across the whole DataLoader batch at once.

    Returns:
        generated_texts: list[str], decoded continuations only (prefix stripped)
        prefix_len: int, width of the forced prefix that was stripped
    """
    batch_size = embedding.shape[0]

    ner_prefix = build_ner_prefix_batch(ner_tokens_list, with_nei)
    bos_col = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    decoder_input_ids = torch.cat([ner_prefix, bos_col], dim=1) if ner_prefix is not None else bos_col

    generated = model.generate(
        embedding.to(device),
        attention_mask.to(device),
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size if beam else 1,
        early_stopping=beam,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prefix_len = decoder_input_ids.shape[1]
    generated_only = generated[:, prefix_len:]
    generated_texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
    return generated_texts, prefix_len


# Evaluate on test set
print('START EVALUATION ...', flush=True)
predictions, references = [], []

cpt = 0
model.eval()
with torch.no_grad():
    for embedding, attention_mask, summary, ner_tokens in test_dataloader:
        cpt += 1
        embedding = embedding.to(device)

        generated_summaries, prefix_len = generate_batch(
            model, embedding, attention_mask, ner_tokens,
            with_nei, args.beam, args.beam_size, args.max_new_tokens,
        )

        predictions.extend(generated_summaries)
        true_summaries = tokenizer.batch_decode(summary[:, 1:].cpu(), skip_special_tokens=True)
        references.extend(true_summaries)

        if with_nei:
            for i, p in enumerate(ner_tokens):
                if p.shape[0] > 1 or (p.shape[0] == 1 and p[0] != -1):
                    ner_decoded = tokenizer.decode([int(t) for t in p], skip_special_tokens=True)
                    print(f"NER TOKENS (sample {i}): {ner_decoded}")

        print(f'BATCH = {cpt}', flush=True)
        for gen, ref in zip(generated_summaries, true_summaries):
            print('GENERATED SUMMARY : ', flush=True)
            print(gen, flush=True)
            print('TRUE SUMMARY : ', flush=True)
            print(ref, flush=True)
            print("#####################################")


# After collecting predictions and references
rouge_scores = rouge.compute(
    predictions=predictions,
    references=references,
    rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL"],
    use_stemmer=True
)

bertscore = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="fr"
)
bertscore_with_rescaling = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="fr",
    rescale_with_baseline=True
)

# ROUGE average scores
mean_rouge_1 = rouge_scores["rouge1"]
mean_rouge_2 = rouge_scores["rouge2"]
mean_rouge_3 = rouge_scores["rouge3"]
mean_rouge_4 = rouge_scores["rouge4"]
mean_rouge_L = rouge_scores["rougeL"]

# BERTScore average scores
mean_bertscore = np.mean(bertscore["f1"])
mean_bertscore_resc = np.mean(bertscore_with_rescaling["f1"])

# Print results
print(f"ROUGE-1: {mean_rouge_1:.4f}")
print(f"ROUGE-2: {mean_rouge_2:.4f}")
print(f"ROUGE-3: {mean_rouge_3:.4f}")
print(f"ROUGE-4: {mean_rouge_4:.4f}")
print(f"ROUGE-L: {mean_rouge_L:.4f}")
print(f"BERTScore F1: {mean_bertscore:.4f}")
print(f"BERTScore F1 with rescaling: {mean_bertscore_resc:.4f}")