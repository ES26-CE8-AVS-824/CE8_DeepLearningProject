from copy import deepcopy
from itertools import islice
import time
import collections

class BPE_Tokenizer:
    def __init__(self):
        self.merges = {}
        self.raw_text = None
        self.byte_text = None
        
    def text_to_byte(self, text):
        text = text.encode('utf-8')
        return list(map(int, text))

    def load_raw(self, vocab_file):
        # Load the raw text data from the specified file
        with open(vocab_file, 'r', encoding='utf-8') as f:
            txt = f.read()
            self.raw_text = txt.replace('\n', ' ')
            self.byte_text = self.text_to_byte(self.raw_text)
        return self.byte_text, self.raw_text
    
    def load_merges(self, merges_file):
        # Load the merges from the specified file
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair, idx = line.strip().rsplit(' ', 1)
                self.merges[tuple(map(int, pair.split()))] = int(idx)
        
    def get_stats(self, text):
        if text is None:
            raise ValueError("Raw text not loaded. Call load_raw() first.")
        # Count the frequency of each byte in the byte text
        return collections.Counter(zip(text, islice(text, 1, None))) 
    
    def merge(self, ids, pair, idx):
        # Merge the specified pair of tokens in the list of token IDs
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
        
    def train(self, vocab_file, iters=10):
        # Train the tokenizer through input text
        self.load_raw(vocab_file)
        t1 = time.time()
        for i in range(iters):
            stats = self.get_stats(self.byte_text)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            self.byte_text = self.merge(self.byte_text, top_pair, idx)
            self.merges[top_pair] = idx
        print(f"Training completed in {time.time() - t1:.2f} seconds.")
        return self.byte_text, self.merges
        

    def encode(self, text):
        # Encode the input
        if isinstance(text, str):
            text = self.text_to_byte(text)
        tokens = deepcopy(text)
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, encoded_tokens):
        # Decode the input
        def single_decode(_id):
            if _id < 256:
                return chr(_id)
            else:
                pair = [k for k, v in self.merges.items() if v == _id][0]
                return single_decode(pair[0]) + single_decode(pair[1])
        return ''.join(single_decode(id) for id in encoded_tokens)
            
            
if __name__ == "__main__":
    Tokenizer = BPE_Tokenizer()
    bt, rt = Tokenizer.train('decoder/test_text.txt', iters=1000)
    # save tokens
    with open('decoder/merges.txt', 'w') as f:
        for pair, idx in Tokenizer.merges.items():
            f.write(f"{pair[0]} {pair[1]} {idx}\n")
    encoded_txt = Tokenizer.encode('Hello, world!')
    print("Encoded:", encoded_txt)
    decoded_txt = Tokenizer.decode(encoded_txt)
    print("Decoded:", decoded_txt)
