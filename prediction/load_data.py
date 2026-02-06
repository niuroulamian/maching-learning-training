import pandas as pd

def load_basic_splits(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv",sep="\t")
    dev   = pd.read_csv(f"{data_dir}/dev.csv",sep="\t")
    test  = pd.read_csv(f"{data_dir}/test.csv",sep="\t")
    return train, dev, test


def load_sequence_data(data_dir):
    seq = pd.read_csv(f"{data_dir}/level_seq.csv",sep="\t")
    meta = pd.read_csv(f"{data_dir}/level_meta.csv",sep="\t")
    return seq, meta
