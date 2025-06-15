import pickle
import regex as re
from collections.abc import Iterable, Iterator
import json
import os


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".encode('utf-8')



class Tokenizer:
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = [sp.encode('utf-8') for sp in special_tokens]
            for sp in special_tokens:
                sp_utf8 = sp.encode('utf-8')
                if sp_utf8 not in self.inverse_vocab:
                    self.inverse_vocab[sp_utf8] = len(self.vocab)
                    self.vocab[len(self.vocab)] = sp_utf8
        else:
            self.special_tokens = None
        self.vocab_size = len(self.vocab)
        self.merges = {} # dict[tuple[bytes, bytes], int]    
        for (a,b) in merges:
            self.merges[a + b] = self.inverse_vocab[a + b] # merge a and b into a new token

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token

    def encode(self, text: str) -> list[int]:
        text = text.encode('utf-8')
        if not self.special_tokens:
            # If no special tokens, just process the whole text
            texts = re.findall(PAT, text)
            return self._encode_texts(texts)
        
        # Create pattern to match any special token
        special_pattern = b'|'.join(re.escape(token) for token in self.special_tokens)
        # Match either special tokens or regular text patterns
        
        parts = []
        current_pos = 0
        for match in re.finditer(special_pattern, text):
            start, end = match.span()
            # Add text before special token
            if start > current_pos:
                parts.append(('text', text[current_pos:start]))
            # Add special token
            parts.append(('special', match.group(0)))
            current_pos = end
        
        # Add remaining text
        if current_pos < len(text):
            parts.append(('text', text[current_pos:]))
        
        # Process all parts
        tokens = []
        for part_type, content in parts:
            if part_type == 'special':
                tokens.append(self.inverse_vocab[content])
            else:
                texts = re.findall(PAT, content)
                tokens.extend(self._encode_texts(texts))
        return tokens
    
    def _encode_texts(self, texts: list[bytes]) -> list[int]:
        """Helper method to encode a list of text segments using byte pair encoding."""
        tokens = []
        for text in texts:
            # Convert text to bytes
            # Initialize with individual bytes
            current_tokens = [self.inverse_vocab[bytes([b])] for b in text]
            
            # Apply BPE merges
            while len(current_tokens) > 1:
                # Find the best merge
                best_merge = None
                best_merge_idx = -1
                
                for i in range(len(current_tokens) - 1):
                    pair = current_tokens[i:i+2]
                    pair_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
                    if pair_bytes in self.merges:
                        if best_merge is None or self.merges[pair_bytes] < best_merge:
                            best_merge = self.merges[pair_bytes]
                            best_merge_idx = i
                
                if best_merge_idx == -1:
                    break
                
                # Apply the best merge
                merge_bytes = self.vocab[current_tokens[best_merge_idx]] + self.vocab[current_tokens[best_merge_idx+1]]
                merged_token = self.merges[merge_bytes]
                current_tokens[best_merge_idx:best_merge_idx+2] = [merged_token]
            
            # Convert final tokens to vocabulary indices
            for token in current_tokens:
                assert isinstance(token, int)
            tokens.extend(current_tokens)
        
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        text_bytes = b''
        for token in ids:
            text_bytes += self.vocab[token]
        text = text_bytes.decode('utf-8', errors='replace')
        return text

# from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
# def get_tokenizer_from_vocab_merges_path(
#     vocab_path: str | os.PathLike,
#     merges_path: str | os.PathLike,
#     special_tokens: list[str] | None = None,
# ):
#     gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
#     with open(vocab_path) as vocab_f:
#         gpt2_vocab = json.load(vocab_f)
#     gpt2_bpe_merges = []
#     with open(merges_path) as f:
#         for line in f:
#             cleaned_line = line.rstrip()
#             if cleaned_line and len(cleaned_line.split(" ")) == 2:
#                 gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
#     # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
#     # just return the original bytes, so we don't force students to use
#     # any particular encoding scheme.
#     vocab = {
#         gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
#         for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
#     }
#     # If any of the special tokens don't exist in the vocab, append them to the vocab.
#     if special_tokens:
#         for special_token in special_tokens:
#             byte_encoded_special_token = special_token.encode("utf-8")
#             if byte_encoded_special_token not in set(vocab.values()):
#                 vocab[len(vocab)] = byte_encoded_special_token

#     merges = [
#         (
#             bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
#             bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
#         )
#         for merge_token_1, merge_token_2 in gpt2_bpe_merges
#     ]
#     return Tokenizer(vocab, merges, special_tokens)

# import tiktoken

# if __name__ == '__main__':
    # tokenizer = Tokenizer.from_files('/root/assignment1-basics/bpe_result/tiny/vocab.pkl', '/root/assignment1-basics/bpe_result/tiny/merge.pkl', ['<s>', '</s>', '<pad>'])
    # print(tokenizer.encode('Hello, world!'))
    # print(tokenizer.decode(tokenizer.encode('Hello, world!')))
    # print(tokenizer.encode('Hello, world! <s>'))
    # print(tokenizer.decode(tokenizer.encode('Hello, world! <s>')))
    # VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    # MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
    
    # tokenizer = get_tokenizer_from_vocab_merges_path(
    #     vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    # )
    # test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # test_string = 's'
    # encoded_ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # # Ensure the special <|endoftext|> token is preserved
    # assert tokenized_string.count("<|endoftext|>") == 3

    # decoded_string = tokenizer.decode(encoded_ids)
    # assert test_string == decoded_string

    # reference_tokenizer = tiktoken.get_encoding("gpt2")
    # tokenizer = get_tokenizer_from_vocab_merges_path(
    #     vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    # )
    # test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    # reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    # ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    # assert tokenizer.decode(ids) == test_string
    # assert reference_tokenizer.decode(reference_ids) == test_string

