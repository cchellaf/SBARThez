from dataset.kaldi_dataset import KaldiDataset, collate_fn
from models.sbarthez_model import SBARThez_BGE
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
import yaml

##################### YAML CONFIG LOADING ##############################

# Load the YAML config
with open("configs/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Accessing values
train_emb_path = config["data"]["train_emb_path"]
train_token_path = config["data"]["train_token_path"]
train_ner_token_path = config["data"]["train_ner_token_path"]

valid_emb_path = config["data"]["valid_emb_path"]
valid_token_path = config["data"]["valid_token_path"]
valid_ner_token_path = config["data"]["valid_ner_token_path"]

batch_size = config["training"]["batch_size"]
num_epochs = config["training"]["num_epochs"]
lr_fc = config["training"]["lr_fc"]
lr_decoder = config["training"]["lr_decoder"]
weight_decay = config["training"]["weight_decay"]

# Whether to prepend the named-entity (NEI) prefix tokens to the decoder
# input/output. If False, training proceeds exactly as if no NER module
# existed (plain summary tokens only). Defaults to True if not set in the
# config, to preserve prior behavior.
with_nei = config["training"].get("with_nei", True)

checkpoint_path = config["model"]["checkpoint_path"]

##################### DATA LOADING ##############################

train_dataset = KaldiDataset(train_emb_path, train_token_path, train_ner_token_path)
val_dataset = KaldiDataset(valid_emb_path, valid_token_path, valid_ner_token_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

##################### MODEL INITIALIZATION ##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBARThez_BGE().to(device)
tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")

fc_params = list(model.fc.parameters())
model_params = list(model.model.parameters())
total_params = fc_params + model_params

optimizer_fc = torch.optim.AdamW(fc_params, lr=lr_fc, weight_decay=weight_decay)  # Larger LR for projection
optimizer_decoder = torch.optim.AdamW(model_params, lr=lr_decoder, weight_decay=weight_decay)  # Smaller LR

scheduler_fc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fc, mode='min', factor=0.5, patience=2)
scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decoder, mode='min', factor=0.5, patience=2)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

scaler = GradScaler()

print(f"Training with NEI module: {with_nei}", flush=True)

##################### TRAINING ##############################
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    print("-------------------------------------")
    print(f"EPOCH : {epoch+1}")

    cpt = 0
    for embedding, attention_mask, summary, ner_tokens in train_dataloader:
        optimizer_fc.zero_grad()
        optimizer_decoder.zero_grad()

        embedding, tgt_input, tgt_output = embedding.to(device), summary[:, :-1].to(device), summary[:, 1:].to(device)

        if with_nei:
            batch_size = tgt_input.shape[0]
            batch_prefixes = ner_tokens

            # Pad and concatenate prefixes to form a tensor of shape (batch_size, prefix_length)
            max_prefix_len = max(p.shape[0] for p in batch_prefixes)
            padded_prefixes = torch.full((batch_size, max_prefix_len), tokenizer.pad_token_id, dtype=torch.long, device=device)

            for i, p in enumerate(batch_prefixes):
                if p.shape[0] == 1 and p[0] == -1:  # Check if the NER tokens are just [-1]
                    cpt += 1

                else : 
                    padded_prefixes[i, : p.shape[0]] = p  # Add prefix only if it's not [-1]

            # Concatenate prefix with tgt_input
            tgt_input = torch.cat([padded_prefixes, tgt_input], dim=1)
            tgt_output = torch.cat([padded_prefixes, tgt_output], dim=1)

        max_len = 1024
        found = False
        for i in range(tgt_input.shape[0]):
            if tgt_input[i].shape[0] > max_len:
                print(f"Skipping sample {i} because length {tgt_input[i].shape[0]} > {max_len}")
                found = True

        if found:
            continue

        with autocast():  # Enables mixed precision
            output = model(embedding, attention_mask.to(device), tgt_input, tgt_output.reshape(-1))
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer_fc)
        scaler.step(optimizer_decoder)
        scaler.update()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Train Loss: {avg_train_loss:.4f}", flush=True)
    
    print(f"TOTAL cpt = {cpt}")

    # Validation step
    model.eval()
    val_loss = 0
    cpt_val = 0
    print('START EVALUATION ...', flush=True)
    predictions, references = [], []

    with torch.no_grad():
        for embedding, attention_mask, summary, ner_tokens in val_dataloader:
            embedding, tgt_input, tgt_output = embedding.to(device), summary[:, :-1].to(device), summary[:, 1:].to(device)

            if with_nei:
                batch_size = tgt_input.shape[0]
                batch_prefixes = ner_tokens

                max_prefix_len = max(p.shape[0] for p in batch_prefixes)
                padded_prefixes = torch.full((batch_size, max_prefix_len), tokenizer.pad_token_id, dtype=torch.long, device=device)

                for i, p in enumerate(batch_prefixes):
                    if p.shape[0] == 1 and p[0] == -1:
                        cpt_val += 1
                    else : 
                        padded_prefixes[i, : p.shape[0]] = p

                tgt_input = torch.cat([padded_prefixes, tgt_input], dim=1)
                tgt_output = torch.cat([padded_prefixes, tgt_output], dim=1)

            with autocast(): 
                output = model(embedding, attention_mask.to(device), tgt_input, tgt_output.reshape(-1))
                loss = output.loss
                
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}", flush=True)
    print(f"TOTAL cpt_val = {cpt_val}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "with_nei": with_nei,
        }, checkpoint_path)
        print(f"✅ New best model saved at {checkpoint_path}", flush=True)

    scheduler_fc.step(avg_val_loss)
    scheduler_decoder.step(avg_val_loss)

print("FINISHED TRAINING !", flush=True)