"""GPT-2 byte BPE, using OpenAI's original pre-tokenization pattern."""
import json
from functools import lru_cache
import regex


class Tokenizer:
    eos = 50256

    def __init__(self, root):
        self.vocab = json.loads((root / "vocab.json").read_text())
        if len(self.vocab) != 50257 or set(self.vocab.values()) != set(range(50257)):
            raise ValueError("expected GPT-2 vocabulary of 50257 tokens")
        self.inverse = {v: k for k, v in self.vocab.items()}
        lines = (root / "merges.txt").read_text().splitlines()
        # Only the first version line is a comment. '# #' and '## ##' are
        # valid GPT-2 merges; dropping them changes IDs and all later ranks.
        merges = [tuple(line.split()) for line in lines[1:] if line]
        if len(merges) != 50000 or any(len(pair) != 2 for pair in merges):
            raise ValueError("expected 50000 GPT-2 BPE merges")
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        mapping = {b: chr(b) for b in printable}
        for index, byte in enumerate(b for b in range(256) if b not in mapping):
            mapping[byte] = chr(256 + index)
        self.byte_encoder = mapping
        self.byte_decoder = {char: byte for byte, char in mapping.items()}
        self.pattern = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @lru_cache(maxsize=8192)
    def bpe(self, token):
        pieces = list(token)
        while len(pieces) > 1:
            pairs = list(zip(pieces, pieces[1:]))
            pair = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            merged, index = [], 0
            while index < len(pieces):
                if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == pair:
                    merged.append(pieces[index] + pieces[index + 1])
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged
        return tuple(pieces)

    def encode(self, text):
        return [self.vocab[piece] for word in self.pattern.findall(text)
                for piece in self.bpe("".join(self.byte_encoder[b] for b in word.encode("utf-8")))]

    def decode_bytes(self, tokens):
        return bytes(self.byte_decoder[c] for token in tokens for c in self.inverse[token])

    def decode(self, tokens):
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")
