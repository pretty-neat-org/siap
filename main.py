import sqlite3
import pandas as pd
from transformers import XLNetForSequenceClassification
import tokenizer as tokenizer
import sentencepiece

if __name__ == '__main__':

    db = sqlite3.connect(r'E:\reddit_dataset\reddit-comments-may-2015\database.sqlite')

    df_data = pd.read_sql_query("SELECT * FROM May2015 LIMIT 50000", db)
    df_data['index'] = df_data.index
    print(df_data.columns)

    print(df_data.head(n=20))

    print(df_data.controversiality.value_counts())

    # It's highly recommended to download bert prtrained model first, then save them into local file
    # In this document, contain confg(txt) and weight(bin) files
    model_file_address = 'model/'

    # Will load config and weight with from_pretrained()
    # Recommand download the model before using
    # Download model from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin"
    # Download model from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json"
    model = XLNetForSequenceClassification.from_pretrained(model_file_address, num_labels=2)

    #
    # max_len = 64
    #
    # full_input_ids = []
    # full_input_masks = []
    # full_segment_ids = []
    #
    # SEG_ID_A = 0
    # SEG_ID_B = 1
    # SEG_ID_CLS = 2
    # SEG_ID_SEP = 3
    # SEG_ID_PAD = 4
    #
    # UNK_ID = tokenizer.encode("<unk>")[0]
    # CLS_ID = tokenizer.encode("<cls>")[0]
    # SEP_ID = tokenizer.encode("<sep>")[0]
    # MASK_ID = tokenizer.encode("<mask>")[0]
    # EOD_ID = tokenizer.encode("<eod>")[0]
    #
    # for i, sentence in enumerate(sentences):
    #     # Tokenize sentence to token id list
    #     tokens_a = tokenizer.encode(sentence)
    #
    #     # Trim the len of text
    #     if (len(tokens_a) > max_len - 2):
    #         tokens_a = tokens_a[:max_len - 2]
    #
    #     tokens = []
    #     segment_ids = []
    #
    #     for token in tokens_a:
    #         tokens.append(token)
    #         segment_ids.append(SEG_ID_A)
    #
    #     # Add <sep> token
    #     tokens.append(SEP_ID)
    #     segment_ids.append(SEG_ID_A)
    #
    #     # Add <cls> token
    #     tokens.append(CLS_ID)
    #     segment_ids.append(SEG_ID_CLS)
    #
    #     input_ids = tokens
    #
    #     # The mask has 0 for real tokens and 1 for padding tokens. Only real
    #     # tokens are attended to.
    #     input_mask = [0] * len(input_ids)
    #
    #     # Zero-pad up to the sequence length at fornt
    #     if len(input_ids) < max_len:
    #         delta_len = max_len - len(input_ids)
    #         input_ids = [0] * delta_len + input_ids
    #         input_mask = [1] * delta_len + input_mask
    #         segment_ids = [SEG_ID_PAD] * delta_len + segment_ids
    #
    #     assert len(input_ids) == max_len
    #     assert len(input_mask) == max_len
    #     assert len(segment_ids) == max_len
    #
    #     full_input_ids.append(input_ids)
    #     full_input_masks.append(input_mask)
    #     full_segment_ids.append(segment_ids)
    #
    #     if 3 > i:
    #         print("No.:%d" % (i))
    #         print("sentence: %s" % (sentence))
    #         print("input_ids:%s" % (input_ids))
    #         print("attention_masks:%s" % (input_mask))
    #         print("segment_ids:%s" % (segment_ids))
    #         print("\n")



