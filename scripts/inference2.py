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
parser.add_argument("--ner", type=str, required=True, help="Path to the NER token .scp file")
parser.add_argument("--beam", action="store_true", help="Use beam search decoding instead of greedy")
parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search decoding")
# --with_nei / --without_nei let the user force the setting explicitly.
# If neither is passed, we auto-detect from the checkpoint (see below).
parser.add_argument("--with_nei", dest="with_nei", action="store_true", default=None,
                     help="Force-enable the NER prefix, overriding whatever the checkpoint says.")
parser.add_argument("--without_nei", dest="with_nei", action="store_false",
                     help="Force-disable the NER prefix, overriding whatever the checkpoint says.")
args = parser.parse_args()

valid_emb_path = args.emb
valid_token_path = args.tok
valid_ner_token_path = args.ner
checkpoint_path = args.ckpt

test_dataset = KaldiDataset(valid_emb_path, valid_token_path, valid_ner_token_path)
test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

##################### MODEL INITIALIZATION ##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBARThez_BGE().to(device)
tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")

checkpoint = torch.load(checkpoint_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_with_nei = checkpoint.get("with_nei", None)
else:
    model.load_state_dict(checkpoint)
    checkpoint_with_nei = None

if args.with_nei is not None:
    # Explicit CLI override takes precedence, but warn if it disagrees
    # with what the checkpoint says it was trained with.
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

# Evaluation Metrics
rouge = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


def build_ner_prefix(ner_tokens, with_nei):
    """
    Returns a (1, prefix_len) LongTensor of real NER tokens on `device`,
    or None if with_nei is False or the NER tokens are the [-1] sentinel
    (i.e. "no entities found" from preprocessing).
    """
    if not with_nei:
        return None

    if isinstance(ner_tokens, list):
        ner_tokens = torch.tensor(ner_tokens, dtype=torch.long)

    if ner_tokens.shape[0] == 1 and ner_tokens[0] == -1:
        return None  # no entities were found for this example

    return ner_tokens.unsqueeze(0).to(device)


def generate_summary_greedy(embedding_sequence, attention_mask, ner_tokens, with_nei, max_length=512):

    # Prepare input token
    input_token = torch.tensor([[tokenizer.bos_token_id]]).to(device)

    ner_prefix = build_ner_prefix(ner_tokens, with_nei)
    if ner_prefix is not None:
        input_token = torch.cat([ner_prefix, input_token], dim=1)  # Concatenate NER tokens

    output_tokens = []

    for i in range(max_length):
        with torch.no_grad():
            output = model(embedding_sequence.to(device), attention_mask.to(device), input_token)
            logits = output.logits[:, -1, :]

        next_token = logits.argmax(dim=-1, keepdim=True)

        output_tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_token = torch.cat([input_token, next_token], dim=-1)    

    return tokenizer.decode(output_tokens, skip_special_tokens=True)

############ BEAM SEARCH IMPLEMENTATION ################
def generate_summary_beam(embedding_sequence, attention_mask, ner_tokens, with_nei, max_length=512, beam_size=3):

    # Prepare input token
    input_token = torch.tensor([[tokenizer.bos_token_id]], device=device)

    ner_prefix = build_ner_prefix(ner_tokens, with_nei)
    num_ner_tokens = 0
    if ner_prefix is not None:
        input_token = torch.cat([ner_prefix, input_token], dim=1)
        num_ner_tokens = ner_prefix.shape[1]

    # Initialize beams
    beams = [(input_token, 0)]  # (sequence, cumulative log probability)
    completed_sequences = []

    for _ in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            with torch.no_grad():
                output = model(embedding_sequence.to(device), attention_mask.to(device), seq)
                logits = output.logits[:, -1, :]

            probs = torch.nn.functional.log_softmax(logits, dim=-1)
            top_k_probs, top_k_tokens = probs.topk(beam_size, dim=-1)
            
            for i in range(beam_size):
                new_token = top_k_tokens[:, i].unsqueeze(0) 
                new_score = score + top_k_probs[:, i].item()
                
                new_seq = torch.cat([seq, new_token], dim=-1) 
                
                if new_token.item() == tokenizer.eos_token_id:
                    completed_sequences.append((new_seq, new_score))
                else:
                    new_beams.append((new_seq, new_score))
        
        # Sort beams by cumulative probability and keep top `beam_size`
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        if len(completed_sequences) >= beam_size:
            break

    if completed_sequences:
        best_sequence = max(completed_sequences, key=lambda x: x[1])[0]
    else:
        best_sequence = max(beams, key=lambda x: x[1])[0]  

    return tokenizer.decode(best_sequence.squeeze(0)[num_ner_tokens:].tolist(), skip_special_tokens=True)


# Evaluate on test set
print('START EVALUATION ...', flush=True)
predictions, references = [], []

rouge_scores_L = []
rouge_scores_1 = []
rouge_scores_2 = []
rouge_scores_3 = []
rouge_scores_4 = []
bertscore_scores = []
bertscore_scores_resc = []

cpt = 0
model.eval()
with torch.no_grad():
    for embedding, attention_mask, summary, ner_tokens in test_dataloader:
        cpt += 1
        embedding = embedding.to(device)

        if args.beam:
            generated_summary = generate_summary_beam(embedding, attention_mask, ner_tokens[0], with_nei, beam_size=args.beam_size)
        else:
            generated_summary = generate_summary_greedy(embedding, attention_mask, ner_tokens[0], with_nei)

        predictions.extend([generated_summary])
        true_summary = tokenizer.batch_decode(summary[:, 1:].cpu(), skip_special_tokens=True)

        if with_nei and (ner_tokens[0].shape[0] > 1 or (ner_tokens[0].shape[0] == 1 and ner_tokens[0][0] != -1)):
            ner_tokens_int = [[int(token) for token in seq] for seq in ner_tokens]  # Convert to integers
            ner_tokens_decoded = tokenizer.batch_decode(ner_tokens_int, skip_special_tokens=True)

            print('NER TOKENS : ')
            print(ner_tokens_decoded)

        references.extend(true_summary)
        print(f'CPT = {cpt}', flush=True)
        print('GENERATED SUMMARY : ', flush=True)
        print(generated_summary, flush=True)
        print('TRUE SUMMARY : ', flush=True)
        print(true_summary, flush=True)
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