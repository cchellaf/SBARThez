import kaldi_io
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")


class KaldiDataset(Dataset):
    def __init__(self, embeddings_scp, tokens_scp, ner_scp=None):
        """
        Args:
            embeddings_scp: Path to SCP file containing embeddings.
            tokens_scp: Path to SCP file containing token sequences.
            ner_scp: Path to SCP file containing named entity token sequences.
        """
        # Load embeddings and tokens
        embeddings = {key: mat for key, mat in kaldi_io.read_mat_scp(embeddings_scp)}
        tokens = {key: [int(x) for x in vec] for key, vec in kaldi_io.read_vec_flt_scp(tokens_scp)}
        if ner_scp is not None:
            ner_tokens = {key: [int(x) for x in vec] for key, vec in kaldi_io.read_vec_flt_scp(ner_scp)}
        else:
            ner_tokens = None

        # Find common keys
        common_keys = set(embeddings.keys()) & set(tokens.keys())
     
        # Keep only the common keys
        self.embeddings = {key: embeddings[key] for key in common_keys}
        self.tokens = {key: tokens[key] for key in common_keys}
        if ner_tokens is not None:
            self.ner_tokens = {key: ner_tokens.get(key, [-1]) for key in common_keys}
        else:
            self.ner_tokens = {key: ner_tokens[key] for key in common_keys}
        self.keys = list(common_keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        embedding = torch.tensor(self.embeddings[key], dtype=torch.float32)
        tokens = torch.tensor(self.tokens[key], dtype=torch.long)
        ner_tokens = torch.tensor(self.ner_tokens[key], dtype=torch.long)
        return embedding, tokens, ner_tokens


def collate_fn(batch):
    embeddings, targets, ner_tokens = zip(*batch)
    embeddings_padded = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (embeddings_padded != 0.0).any(dim=-1).float()
    return embeddings_padded, attention_mask, targets_padded, ner_tokens
