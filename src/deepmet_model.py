import torch
import torch.nn as nn
from transformers import RobertaModel


class DeepMet(nn.Module):

    def __init__(self, num_tokens, num_encoder_inputs, num_encoder_heads, num_encoder_layers, num_encoder_hidden_states, dropout_rate):
        super(DeepMet, self).__init__()

        self.embedding_layer = RobertaModel.from_pretrained('roberta-base')

        # Use the same transformer encoder for both, since we want to use the same weights
        encoder_layers = nn.TransformerEncoderLayer(d_model=num_encoder_inputs, nhead=num_encoder_heads,
                                                    dim_feedforward=num_encoder_hidden_states, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        self.average_pool_a = nn.AvgPool1d(kernel_size=num_tokens)
        self.average_pool_b = nn.AvgPool1d(kernel_size=num_tokens)

        # Metaphor discrimination layer
        self.dropout = nn.Dropout(dropout_rate)
        self.discriminator = nn.Linear(2 * num_encoder_hidden_states, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids,
                local_attention_mask, local_token_type_ids):

        # Embedding
        global_intermediate = self.embedding_layer(global_input_ids, global_attention_mask, global_token_type_ids)
        local_intermediate = self.embedding_layer(local_input_ids, local_attention_mask, local_token_type_ids)

        # Transformer encoder
        global_intermediate = self.transformer_encoder(global_intermediate.last_hidden_state)
        local_intermediate = self.transformer_encoder(local_intermediate.last_hidden_state)

        # Average pooling
        global_intermediate = global_intermediate.permute(0, 2, 1)
        local_intermediate = local_intermediate.permute(0, 2, 1)
        global_intermediate = self.average_pool_a(global_intermediate).squeeze(dim=2)
        local_intermediate = self.average_pool_b(local_intermediate).squeeze(dim=2)

        # Metaphor discrimination
        out = torch.cat((global_intermediate, local_intermediate), dim=1)
        out = self.dropout(out)
        out = self.discriminator(out)
        out = self.softmax(out)

        return out
