import torch
import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activation(outputs)
        return outputs

class UnanswerableClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pooler = Pooler(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        self.critieron = nn.CrossEntropyLoss()
    
    #(batch hidden), (batch)
    def forward(self, final_cls, target):
        pooled_cls = self.pooler(final_cls)
        logits = self.classifier(pooled_cls)

        #0: answerable 1: unanswerable
        loss = self.critieron(logits, target)
        #scalar, (batch)
        return loss, logits[:, 1]

class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.projection_out = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, inputs):
        outputs = self.projection_in(inputs)
        outputs = self.activation(outputs)
        outputs = self.projection_out(outputs)
        return outputs


class CrossAttentionForQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.cross_attention = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=config.attention_probs_dropout_prob)
        
        self.feed_forward = Intermediate(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        

    #(batch sequence hidden), (batch, sequence), (batch sequence)
    def forward(self, final_hidden_state, attention_mask, token_type_ids):
        question_mask = ((token_type_ids == 1) | (attention_mask == 0))
        question_mask[:, 0] = True
        

        
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, attention_mask.size(1))
        attention_mask = attention_mask & attention_mask.transpose(1, 2)
        attention_mask = attention_mask.float()
        attention_mask = torch.repeat_interleave(attention_mask, self.num_heads, dim=0)
    

        h = final_hidden_state.transpose(0, 1)

        #Q(T N H), K(S N H), V(S N H)
        attn_output, attn_output_weights = self.cross_attention(h, h, h, key_padding_mask=question_mask, attn_mask=attention_mask)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.dropout(attn_output)
        attn_output = self.layernorm(attn_output + final_hidden_state)
        h_prime = self.feed_forward(attn_output)
        h_prime = self.dropout(h_prime)
        h_prime = self.layernorm(h_prime + attn_output)
        #(batch sequence hidden)
        return h_prime
