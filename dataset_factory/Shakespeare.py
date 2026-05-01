# /datasets/SHAKESPEARE.py

import os
import re
import json
import urllib.request
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from .base import DatasetGenerator

# Class name is SHAKESPEARE_Generator as requested
class SHAKESPEARE_Generator(DatasetGenerator):
    """
    This definitive version implements the official LEAF benchmark's preprocessing
    logic for the Shakespeare dataset. It uses a hardcoded list of plays and a
    robust parser to correctly extract speaker dialogue.
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
        seq_len: int = 80,
        min_samples_per_client: int = 20,
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
        self.seq_len = seq_len
        self.min_samples_per_client = min_samples_per_client

    # === COMPLETELY REWRITTEN PARSER BASED ON OFFICIAL LEAF SCRIPT ===
    def download_and_process_raw_data(self) -> dict[str, str]:
        """
        Downloads raw text from Gutenberg and parses it using the official LEAF
        benchmark's methodology.
        """
        raw_url = "https://www.gutenberg.org/files/100/100-0.txt"
        txt_path = os.path.join(self.rawdata_path, "shakespeare.txt")

        if not os.path.exists(txt_path):
            print("Downloading Complete Works of Shakespeare from Gutenberg...")
            os.makedirs(self.rawdata_path, exist_ok=True)
            urllib.request.urlretrieve(raw_url, txt_path)

        print("Processing raw Shakespeare text using official LEAF methodology...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        ### <<< START: Logic adapted from official LEAF 'shake_utils.py' >>> ###
        
        # Hardcoded list of plays, the key to reliable parsing
        plays = [
            "The Tragedy of Antony and Cleopatra", "The Tragedy of Coriolanus",
            "The Tragedy of Hamlet, Prince of Denmark", "The Tragedy of Julius Caesar",
            "The Tragedy of King Lear", "The Tragedy of Macbeth", "The Tragedy of Othello, the Moor of Venice",
            "The Tragedy of Romeo and Juliet", "The Tragedy of Titus Andronicus",
            "The Life of Timon of Athens", "The History of Troilus and Cressida",
            "All's Well That Ends Well", "As You Like It", "The Comedy of Errors",
            "Cymbeline", "Love's Labour's Lost", "Measure for Measure", "The Merry Wives of Windsor",
            "The Merchant of Venice", "A Midsummer Night's Dream", "Much Ado About Nothing",
            "Pericles, Prince of Tyre", "The Taming of the Shrew", "The Tempest",
            "Twelfth Night; Or, What You Will", "The Two Gentlemen of Verona", "The Winter's Tale",
            "The Life and Death of King John", "The Tragedy of King Richard the Second",
            "The First Part of King Henry the Fourth", "The Second Part of King Henry the Fourth",
            "The Life of King Henry the Fifth", "The First Part of King Henry the Sixth",
            "The Second Part of King Henry the Sixth", "The Third Part of King Henry the Sixth",
            "The Tragedy of King Richard the Third", "The Famous History of the Life of King Henry the Eighth"
        ]

        # Find the start of the first play to ignore pre-content
        text = raw_text
        for i, play in enumerate(plays):
            if play.upper() in text:
                text = text[text.find(play.upper()):]
                plays = plays[i:]
                break
        
        # Split text by plays
        for i, play in enumerate(plays[:-1]):
            next_play = plays[i+1]
            text = text.replace(next_play.upper(), "$$$")
        text = text.split("$$$")

        speakers_dialogue = defaultdict(str)
        # Parse each play
        for i, play_text in enumerate(text):
            lines = play_text.split('\n')
            current_speaker = ''
            for line in lines:
                # This is the speaker identification rule from the official script
                if re.match(r'^[A-Z][A-Z\s\.]+$', line.strip()) and len(line.strip()) < 25:
                    current_speaker = line.strip()
                elif current_speaker and line.strip() != '' and not re.match(r'^\[.*\]$', line.strip()):
                    # Line is dialogue
                    dialogue = line.strip().lower()
                    speakers_dialogue[current_speaker] += dialogue + ' '

        ### <<< END: Logic adapted from official LEAF 'shake_utils.py' >>> ###

        # Filter out speakers with too little dialogue
        min_char_len = self.seq_len + 1
        filtered_dialogue = {
            speaker: text.strip()
            for speaker, text in speakers_dialogue.items()
            if len(text) >= min_char_len
        }
        return filtered_dialogue

    def generate_data(self):
        if self.check():
            return

        dialogue_by_speaker = self.download_and_process_raw_data()
        
        all_speakers = sorted(dialogue_by_speaker.keys())
        if len(all_speakers) < self.num_nodes:
            raise ValueError(
                f"Requested {self.num_nodes} nodes, but only {len(all_speakers)} "
                f"speakers with enough dialogue (>{self.seq_len + 1} chars) were found. "
                f"Try reducing `num_nodes`."
            )
        
        speakers_with_enough_samples = [
            s for s in all_speakers 
            if (len(dialogue_by_speaker[s]) - self.seq_len) >= self.min_samples_per_client
        ]
        
        if len(speakers_with_enough_samples) < self.num_nodes:
             raise ValueError(
                f"Requested {self.num_nodes} nodes, but only {len(speakers_with_enough_samples)} "
                f"have the required number of samples (min_samples_per_client={self.min_samples_per_client}). "
                f"Try reducing `num_nodes` or `min_samples_per_client`."
            )

        selected_speakers = np.random.choice(speakers_with_enough_samples, self.num_nodes, replace=False)

        full_text = "".join(dialogue_by_speaker[s] for s in selected_speakers)
        all_chars = sorted(list(set(full_text)))
        char_to_idx = {char: i + 1 for i, char in enumerate(all_chars)}
        char_to_idx['<pad>'] = 0
        self.num_classes = len(char_to_idx)
        self.vocab_size = len(char_to_idx)
        self.num_classes = self.vocab_size
        vocab_path = os.path.join(self.dir_path, "vocab.json")
        with open(vocab_path, 'w') as f:
            json.dump(char_to_idx, f)

        train_data_nodes, test_data_nodes = [], []
        statistic = [defaultdict(int) for _ in range(self.num_nodes)]

        for i, speaker in enumerate(selected_speakers):
            text = dialogue_by_speaker[speaker]
            
            sequences, targets = [], []
            if len(text) > self.seq_len:
                for j in range(len(text) - self.seq_len):
                    sequences.append(text[j:j + self.seq_len])
                    targets.append(text[j + self.seq_len])

            if not sequences:
                train_data_nodes.append({'x': np.array([])})
                test_data_nodes.append({'x': np.array([])})
                continue
            
            seq_train, seq_test, y_train, y_test = train_test_split(sequences, targets, train_size=self.train_ratio, random_state=42)

            x_train_idx = np.array([[char_to_idx.get(c, 0) for c in s] for s in seq_train], dtype=np.int64)
            y_train_idx = np.array([char_to_idx.get(c, 0) for c in y_train], dtype=np.int64)
            train_data_nodes.append({'x': x_train_idx, 'y': y_train_idx})
            
            x_test_idx = np.array([[char_to_idx.get(c, 0) for c in s] for s in seq_test], dtype=np.int64)
            y_test_idx = np.array([char_to_idx.get(c, 0) for c in y_test], dtype=np.int64)
            test_data_nodes.append({'x': x_test_idx, 'y': y_test_idx})

            for char_idx in np.concatenate([y_train_idx, y_test_idx]):
                statistic[i][int(char_idx)] += 1
        
        statistic_final = [sorted(list(client_stats.items())) for client_stats in statistic]
        print(f"Number of classes (unique characters): {self.num_classes}")
        print(f"Number of nodes (speakers) found and processed: {len(selected_speakers)}")
        self.plot(statistic=statistic_final, num_classes=self.num_classes)
        self.save_file(
                    train_data_nodes, 
                    test_data_nodes, 
                    self.num_classes, 
                    statistic_final,
                    vocab_size=self.vocab_size # Add this keyword argument
                )
        all_test_x = np.concatenate([data['x'] for data in test_data_nodes if data['x'].shape[0] > 0])
        all_test_y = np.concatenate([data['y'] for data in test_data_nodes if data['y'].shape[0] > 0])
        test_generalization = {'x': all_test_x, 'y': all_test_y}
        with open(self.test_path + "server.npz", "wb") as f:
            np.savez_compressed(f, data=test_generalization)