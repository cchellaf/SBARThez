import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM


class SBARThez_BGE(nn.Module):
    model_name = "moussaKam/barthez"
    def __init__(self, model_name=model_name):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if hasattr(self.model.model.encoder, 'embed_tokens'):
            del self.model.model.encoder.embed_tokens

        self.fc = nn.Linear(1024, 768)
        self.activation = nn.GELU() 

    def forward(self, embeddings, attention_mask, decoder_input_ids, labels=None):
        input_embeddings = self.fc(embeddings) 
        input_embeddings = self.activation(input_embeddings) 
        logits =  self.model(
            inputs_embeds=input_embeddings,  # Pass precomputed embeddings
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return logits

class SBARThez_SONAR(nn.Module):
    model_name = "moussaKam/barthez"
    def __init__(self, model_name=model_name):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if hasattr(self.model.model.encoder, 'embed_tokens'):
            del self.model.model.encoder.embed_tokens

        self.fc = nn.Linear(1024, 768)
        self.activation = nn.GELU() 

    def forward(self, embeddings, attention_mask, decoder_input_ids, labels=None):
        input_embeddings = self.fc(embeddings) 
        input_embeddings = self.activation(input_embeddings) 
        logits =  self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return logits
    
class SBARThez_LaBSE(nn.Module):
    model_name = "moussaKam/barthez"
    def __init__(self, model_name=model_name, dropout_rate=0.1):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if hasattr(self.model.model.encoder, 'embed_tokens'):
            del self.model.model.encoder.embed_tokens

        self.activation = nn.GELU() 

    def forward(self, embeddings, attention_mask, decoder_input_ids, labels=None):
        input_embeddings = self.activation(embeddings) 
        logits =  self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return logits
