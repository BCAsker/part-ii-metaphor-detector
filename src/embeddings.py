import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

batch_size = 64


def prepare_inputs(df):
    global_context_input = df['sentence'] + " </s> " + df['query'] + " </s> " + df['pos'] + " </s> " + df['fgpos']
    local_context_input = df['local'] + " </s> " + df['query'] + " </s> " + df['pos'] + " </s> " + df['fgpos']

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # We've added the intermediate separator tokens, but let the tokenizer handle the start, end and padding tokens
    global_context_tokenized = tokenizer(list(global_context_input),
                                         add_special_tokens=True, padding=True, return_tensors='pt')
    local_context_tokenized = tokenizer(list(local_context_input),
                                        add_special_tokens=True, padding=True, return_tensors='pt')

    global_token_type_ids = torch.zeros(global_context_tokenized['input_ids'].size(), dtype=torch.long)
    local_token_type_ids = torch.zeros(local_context_tokenized['input_ids'].size(), dtype=torch.long)

    return global_context_tokenized['input_ids'], global_context_tokenized['attention_mask'], global_token_type_ids, \
        local_context_tokenized['input_ids'], local_context_tokenized['attention_mask'], local_token_type_ids,


# A simple test to make sure we've prepared our inputs correctly and that we get some form of embeddings out, in the
# correct shape
def apply_to_model(input_data):

    global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, \
        local_token_type_ids = input_data

    print("global_input_ids")
    print(global_input_ids)
    print()
    print("local_token_type_ids")
    print(local_token_type_ids)
    print()

    model = RobertaModel.from_pretrained('roberta-base')

    print("Device test before .to():" + str(global_input_ids.device))
    global_input_ids.to(torch.device(0))
    print("Device test after .to():" + str(global_input_ids.device))

    if torch.cuda.is_available():
        global_input_ids = global_input_ids.to(torch.device(0))
        global_attention_mask = global_attention_mask.to(torch.device(0))
        global_token_type_ids = global_token_type_ids.to(torch.device(0))
        local_input_ids = local_input_ids.to(torch.device(0))
        local_attention_mask = local_attention_mask.to(torch.device(0))
        local_token_type_ids = local_token_type_ids.to(torch.device(0))
        model.to(torch.device(0))

    model.eval()

    num_sentences = len(df_train_compact)
    batches = num_sentences // batch_size + 1  # Make sure we don't forget the last few sentences

    with torch.no_grad():
        for batch_num in range(batches):

            first = batch_num * batch_size
            last = min((batch_num + 1) * batch_size, num_sentences)

            global_outputs = model(input_ids=global_input_ids[first:last],
                                   attention_mask=global_attention_mask[first:last],
                                   token_type_ids=global_token_type_ids[first:last],
                                   output_hidden_states=True)  # position_ids done automatically

            local_outputs = model(input_ids=local_input_ids[first:last],
                                  attention_mask=local_attention_mask[first:last],
                                  token_type_ids=local_token_type_ids[first:last],
                                  output_hidden_states=True)  # position_ids done automatically

            last_hidden_state_global = global_outputs.last_hidden_state
            print("G, Batch: " + str(batch_num) + ", Slice: " + str(batch_num * batch_size) + ":" +
                  str(min((batch_num + 1) * batch_size, num_sentences)) + ", Size: " +
                  str(last_hidden_state_global.shape))

            last_hidden_state_local = local_outputs.last_hidden_state
            print("L, Batch: " + str(batch_num) + ", Slice: " + str(batch_num * batch_size) + ":" +
                  str(min((batch_num + 1) * batch_size, num_sentences)) + ", Size: " +
                  str(last_hidden_state_local.shape))

            print()


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

    inputs = prepare_inputs(df_train)
    apply_to_model(inputs)

