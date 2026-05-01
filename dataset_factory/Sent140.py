# /datasets/SENT140.py

import os
import re
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import DatasetGenerator


class SENT140_Generator(DatasetGenerator):
    """
    This definitive version re-implements the official LEAF preprocessing pipeline
    in Python. It downloads the original raw data from Stanford, processes it,
    and partitions it by user.
    """
    def __init__(
        self,
        num_nodes: int,
        dataset_name: str,
        batch_size: int,
        train_ratio: float,
        alpha: float,
        niid: bool,
        balance: bool,
        partition: str,
        class_per_client: int,
        plot_ylabel_step: int,
        vocab_size: int = 10000,
        max_seq_len: int = 25,
        min_samples_per_client: int = 10,
    ):
        super().__init__(
            num_nodes=num_nodes,
            dataset_name=dataset_name,
            batch_size=batch_size,
            train_ratio=train_ratio,
            alpha=alpha,
            niid=niid,
            balance=balance,
            partition="pre",
            class_per_client=class_per_client,
            plot_ylabel_step=plot_ylabel_step,
        )
        # Store the user-defined limit, but the true vocab size will be calculated.
        self.vocab_size_limit = vocab_size
        self.max_seq_len = max_seq_len
        self.min_samples_per_client = min_samples_per_client

    def download_and_process_raw_data(self) -> pd.DataFrame:
        """
        Downloads the original source zip, extracts, and processes the raw CSV.
        """
        raw_zip_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        zip_path = os.path.join(self.rawdata_path, "sent140_raw.zip")
        csv_path = os.path.join(self.rawdata_path, "training.1600000.processed.noemoticon.csv")

        if not os.path.exists(csv_path):
            print("Downloading original Sentiment140 dataset from cs.stanford.edu...")
            os.makedirs(self.rawdata_path, exist_ok=True)
            urllib.request.urlretrieve(raw_zip_url, zip_path)

            print("Extracting raw CSV...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extracts to 'training.1600000.processed.noemoticon.csv'
                zip_ref.extractall(self.rawdata_path)
            os.remove(zip_path)

        print("Processing raw CSV file...")
        col_names = ['label', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=col_names)
        
        df = df[df['label'] != 2]
        df['label'] = df['label'].apply(lambda x: 1 if x == 4 else 0)

        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= self.min_samples_per_client].index
        df = df[df['user'].isin(valid_users)]
        
        return df

    def _build_vocab(self, all_sentences: list[str]) -> dict[str, int]:
        word_counts = Counter(word for sent in all_sentences for word in sent.split())
        most_common = word_counts.most_common(self.vocab_size_limit - 2)
        word_to_idx = {'<pad>': 0, '<unk>': 1}
        for i, (word, _) in enumerate(most_common):
            word_to_idx[word] = i + 2
        return word_to_idx

    def _tokenize_and_pad(self, sentence: str, word_to_idx: dict[str, int]) -> list[int]:
        seq = [word_to_idx.get(word, word_to_idx['<unk>']) for word in sentence.split()]
        if len(seq) < self.max_seq_len:
            seq.extend([word_to_idx['<pad>']] * (self.max_seq_len - len(seq)))
        else:
            seq = seq[:self.max_seq_len]
        return seq

    def generate_data(self):
        if self.check():
            return

        df = self.download_and_process_raw_data()
        all_users = df['user'].unique()

        if len(all_users) < self.num_nodes:
             raise ValueError(f"Requested {self.num_nodes} nodes, but only {len(all_users)} with >= {self.min_samples_per_client} samples are available.")
        
        selected_users = np.random.choice(all_users, self.num_nodes, replace=False)
        df = df[df['user'].isin(selected_users)]

        word_to_idx = self._build_vocab(df['text'].tolist())

        # === THE FIX IS HERE ===
        # Calculate the TRUE vocab size and use it consistently.
        self.vocab_size = len(word_to_idx)
        print(f"SENT140: Generated vocabulary with actual size: {self.vocab_size}")

        vocab_path = os.path.join(self.dir_path, "vocab.json")
        with open(vocab_path, 'w') as f:
            json.dump(word_to_idx, f)
        
        train_data_nodes = []
        test_data_nodes = []
        statistic = [defaultdict(int) for _ in range(self.num_nodes)]

        for i, user_id in enumerate(selected_users):
            user_df = df[df['user'] == user_id]
            X = user_df['text'].tolist()
            y = user_df['label'].tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_ratio, random_state=42)

            train_x_tokenized = [self._tokenize_and_pad(s, word_to_idx) for s in X_train]
            train_data_nodes.append({'x': np.array(train_x_tokenized, dtype=np.int64), 'y': np.array(y_train, dtype=np.int64)})
            
            test_x_tokenized = [self._tokenize_and_pad(s, word_to_idx) for s in X_test]
            test_data_nodes.append({'x': np.array(test_x_tokenized, dtype=np.int64), 'y': np.array(y_test, dtype=np.int64)})

            for label in y:
                statistic[i][label] += 1
        
        statistic_final = [sorted(list(client_stats.items())) for client_stats in statistic]
        self.num_classes = 2
        self.plot(statistic=statistic_final, num_classes=self.num_classes)
        
        # === AND THE FIX IS HERE ===
        # Pass the true vocab_size to the save function.
        self.save_file(
            train_data=train_data_nodes, 
            test_data=test_data_nodes, 
            num_classes=self.num_classes, 
            statistic=statistic_final,
            vocab_size=self.vocab_size
        )

        all_test_x = np.concatenate([data['x'] for data in test_data_nodes if data['x'].shape[0] > 0])
        all_test_y = np.concatenate([data['y'] for data in test_data_nodes if data['y'].shape[0] > 0])
        test_generalization = {'x': all_test_x, 'y': all_test_y}
        with open(self.test_path + "server.npz", "wb") as f:
            np.savez_compressed(f, data=test_generalization)