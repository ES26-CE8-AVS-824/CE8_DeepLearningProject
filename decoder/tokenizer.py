class BPD_Tokenizer:
    def __init__(self):
        self.merges = {}
        self.raw_text = None
        self.byte_text = None
        self.merges = {}

    def load_raw(self, vocab_file):
        # Load the raw text data from the specified file
        with open(vocab_file, 'r', encoding='utf-8') as f:
            txt = f.read()
            self.raw_text = txt.replace('\n', ' ')
            self.byte_text = list(map(int, self.raw_text))
        
    def get_stats(self, text):
        if text is None:
            raise ValueError("Raw text not loaded. Call load_raw() first.")
        # Count the frequency of each byte in the byte text
        counts = {}
        for pair in zip(text, text[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(ids, pair, idx):
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
        for i in range(iters):
            stats = self.get_stats(self.byte_text)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            self.byte_text = self.merge(self.byte_text, top_pair, idx)
            self.merges[top_pair] = idx
        

    def encode(self, iters=10):
        tokens = self.byte_text.copy()
        for _ in range(iters):
            stats = self.get_stats()
            if not stats:
                break
            max_pair = max(stats, key=stats.get)
            self.merges[max_pair] = len(self.vocab)
            tokens = [self.merges.get((tokens[i], tokens[i+1]), tokens[i]) for i in range(len(tokens)-1)]
        return tokens

    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(token_id, '[UNK]') for token_id in token_ids]
        return ' '.join(tokens)