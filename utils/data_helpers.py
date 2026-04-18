import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import six


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def get_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters.
       Read configuration information from json configuration file"""
    dict = {}
    with open(json_file, 'r') as reader:
        text = reader.read()
    json_file = json.loads(text)
    for (key, value) in six.iteritems(json_file):
        dict[key] = value
    return dict


class LoadMultiPeptideClassificationDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=1,
                 is_sample_shuffle=True,
                 seed=42
                 ):
        
        if tokenizer is None:
            raise ValueError("An EsmTokenizer must be passed to `tokenizer`.")
            
        self.tokenizer = tokenizer
        self.PAD_IDX = pad_index
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle
        

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def data_process(self, filepath):
        """
        Extracts raw sequences and labels from the .npy dictionary.
        Tokenization is deferred to the generate_batch stage.
        """
        seq_label_dict = np.load(filepath, allow_pickle=True).item()
        print(f"Loaded {len(seq_label_dict.keys())} samples from {filepath}")
        
        data = list()
        max_len = 0
        
        for seq in seq_label_dict.keys():
            original_sequence = seq 
            
            # Process labels
            l = seq_label_dict[seq]
            l = l.astype(float) 
            
            # Approximate max length tracking (+2 for CLS/EOS tokens)
            max_len = max(max_len, len(original_sequence) + 2)
            
            # Store raw sequence and label
            data.append((original_sequence, l))
            
        return data, max_len

    def load_train_test_data(self, train_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        
        test_data, _ = self.data_process(filepath=test_file_path)
        test_iter = DataLoader(
            test_data, 
            batch_size=self.batch_size,
            shuffle=False, 
            collate_fn=self.generate_batch,
            worker_init_fn=seed_worker,
            generator=self.generator
        )
        if only_test:
            return test_iter
            
        train_data, max_sen_len = self.data_process(filepath=train_file_path)
        
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
            
        train_iter = DataLoader(
            train_data, 
            batch_size=self.batch_size,  
            shuffle=self.is_sample_shuffle, 
            collate_fn=self.generate_batch,
            worker_init_fn=seed_worker,
            generator=self.generator
        )
                                
        return train_iter, test_iter

    def generate_batch(self, data_batch):
        """
        Dynamically tokenizes a batch of sequences using the ESM Tokenizer.
        """
        batch_label, batch_sequences = [], [] 
        
        for (seq, label) in data_batch: 
            batch_sequences.append(seq) 
            batch_label.append(label.tolist() if isinstance(label, np.ndarray) else label)
            
        # Use ESM tokenizer to handle all encoding, padding, and truncation automatically
        if self.max_sen_len is not None and self.max_sen_len != 'same':
            encoded = self.tokenizer(
                batch_sequences,
                padding='max_length',
                truncation=True,
                max_length=self.max_sen_len,
                return_tensors='pt'
            )
        else:
            encoded = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
        # encoded['input_ids'] is shape [batch_size, sequence_length].
        # Transpose it to [sequence_length, batch_size] to match your training code's expectation.
        batch_sentence = encoded['input_ids'].transpose(0, 1)
        
        batch_label = torch.tensor(batch_label, dtype=torch.float)
        
        # Return 3 values: Token IDs, Labels, Raw Sequences
        return batch_sentence, batch_label, batch_sequences