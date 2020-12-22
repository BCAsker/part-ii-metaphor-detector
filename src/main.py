import pandas as pd
import numpy as np
import torch
import os
import embeddings
import deepmet_model


def save_prepared(global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids):
    torch.save(global_input_ids, '../data/intermediate/global_input_ids.pt')
    torch.save(global_attention_mask, '../data/intermediate/global_attention_mask.pt')
    torch.save(global_token_type_ids, '../data/intermediate/global_token_type_ids.pt')
    torch.save(local_input_ids, '../data/intermediate/local_input_ids.pt')
    torch.save(local_attention_mask, '../data/intermediate/local_attention_mask.pt')
    torch.save(local_token_type_ids, '../data/intermediate/local_token_type_ids.pt')


def load_prepared():
    global_input_ids = torch.load('../data/intermediate/global_input_ids.pt')
    global_attention_mask = torch.load('../data/intermediate/global_attention_mask.pt')
    global_token_type_ids = torch.load('../data/intermediate/global_token_type_ids.pt')
    local_input_ids = torch.load('../data/intermediate/local_input_ids.pt')
    local_attention_mask = torch.load('../data/intermediate/local_attention_mask.pt')
    local_token_type_ids = torch.load('../data/intermediate/local_token_type_ids.pt')
    return global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids


def check_prepared():
    names = ['global_input_ids.pt', 'global_attention_mask.pt', 'global_token_type_ids.pt', 'local_input_ids.pt', 'local_attention_mask.pt', 'local_token_type_ids.pt']
    return np.all([os.path.exists('../data/intermediate/' + name) for name in names])


if __name__ == '__main__':
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.width", 0)

    # Load the dataframes containing the raw inputs for our embedding layer
    df_vua_train = pd.read_csv('../data/VUA/vua_train_tokenized.csv', index_col='token_id').dropna()
    df_toefl_train = pd.read_csv('../data/TOEFL/toefl_train_tokenized.csv', index_col='token_id').dropna()
    df_vua_train_compact = pd.read_csv('../data/VUA/vua_train_tokenized_compact.csv', index_col='sentence_id', sep=';',
                                       error_bad_lines=False).dropna()
    df_toefl_train_compact = pd.read_csv('../data/TOEFl/toefl_train_tokenized_compact.csv', index_col='sentence_id',
                                         sep=';', error_bad_lines=False).dropna()
    df_train_compact = pd.concat([df_vua_train_compact, df_toefl_train_compact])

    # Convert the strings back into lists
    df_train_compact['tokens'] = df_train_compact['tokens'].apply(lambda s: s[1:-1].split(', '))
    df_train_compact['metaphors'] = df_train_compact['metaphors'].apply(lambda s: [int(i) for i in s[1:-1].split(', ')])
    df_train_compact['pos'] = df_train_compact['pos'].apply(lambda s: [str(i) for i in s[1:-1].split(', ')])
    df_train_compact['fgpos'] = df_train_compact['fgpos'].apply(lambda s: [str(i) for i in s[1:-1].split(', ')])
    df_train = pd.concat([df_vua_train, df_toefl_train])

    if check_prepared():
        prepared = load_prepared()
    else:
        prepared = embeddings.prepare_inputs(df_train)
        save_prepared(*prepared)

    num_tokens = len(prepared[0][0])
    model = deepmet_model.DeepMet(num_tokens=num_tokens,
                                  num_encoder_inputs=768,
                                  num_encoder_heads=2,
                                  num_encoder_layers=2,
                                  num_encoder_hidden_states=768,
                                  dropout_rate=0.2)

    print("Model loaded, inputs prepared")

    prepared = list(prepared)
    if torch.cuda.is_available():
        for i in range(len(prepared)):
            prepared[i] = (prepared[i]).to(torch.device(0))
        model.to(torch.device(0))

    model.eval()

    print("Launching model")

    with torch.no_grad():
        out = model.forward(prepared[0][0:3], prepared[1][0:3], prepared[2][0:3], prepared[3][0:3], prepared[4][0:3], prepared[5][0:3])

    print(out)

    print("Done!")
