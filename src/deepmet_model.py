import torch
import torch.nn as nn
from transformers import RobertaModel


class DeepMet(nn.Module):

    def __init__(self, num_tokens, dropout_rate):
        super(DeepMet, self).__init__()

        # Use the same transformer encoder for both, since we want to use the same weights
        self.embedding_and_transformer_layers = RobertaModel.from_pretrained('roberta-base')

        self.average_pool_b = nn.AvgPool1d(kernel_size=num_tokens)

        # Metaphor discrimination layer
        self.dropout = nn.Dropout(dropout_rate)
        self.discriminator = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, local_input_ids, local_attention_mask, local_token_type_ids):

        # Embedding and transformer encoder
        local_intermediate = self.embedding_and_transformer_layers(local_input_ids, local_attention_mask,
                                                                   local_token_type_ids)

        # Average pooling
        local_intermediate = local_intermediate.last_hidden_state.permute(0, 2, 1)
        local_intermediate = self.average_pool_b(local_intermediate).squeeze(dim=2)

        # Metaphor discrimination
        out = self.dropout(local_intermediate)
        out = self.discriminator(out)
        out = self.softmax(out)

        return out
