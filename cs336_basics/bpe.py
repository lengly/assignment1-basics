import regex as re
import heapq
from dataclasses import dataclass, field
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import pickle
from multiprocessing import Process, Queue

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass
class PairValue:
    freq : int = 0
    pos_idx : set[int] = field(default_factory=set)

class ByteWord:
    def __init__(self, word: str, weight: int, id: int):
        self.word = word
        self.weight = weight
        self.id = id

    def encode(self, pair_hash: dict[tuple[int, int] : tuple[int, list[int]]]):
        self.tokens = list(self.word.encode('utf-8'))
        if len(self.tokens) <= 1:
            return
        for key in zip(self.tokens[:-1], self.tokens[1:]):
            if key not in pair_hash:
                pair_hash[key] = PairValue(freq=0, pos_idx=set())
            pair_hash[key].freq += self.weight
            pair_hash[key].pos_idx.add(self.id)

    def __repr__(self):
        return f"({self.id}, {self.word}, {self.weight})"

    def merge(self, merge_key: tuple[int, int], new_id: int, pair_hash: dict[tuple[int, int] : tuple[int, list[int]]]):
        for key in zip(self.tokens[:-1], self.tokens[1:]):
            if key not in pair_hash:
                pair_hash[key] = PairValue(freq=0, pos_idx=set())
            pair_hash[key].freq -= self.weight
        new_tokens = []
        i = 0
        while i < len(self.tokens):
            if i < len(self.tokens) - 1 and self.tokens[i] == merge_key[0] and self.tokens[i+1] == merge_key[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1
        self.tokens = new_tokens
        new_pairs = set()
        for key in zip(self.tokens[:-1], self.tokens[1:]):
            if key not in pair_hash:
                pair_hash[key] = PairValue(freq=0, pos_idx=set())
                new_pairs.add(key)
            pair_hash[key].freq += self.weight
            pair_hash[key].pos_idx.add(self.id)
        return new_pairs

class QueueItem:
    def __init__(self, freq: int, key_int: tuple[int, int], key_bytes: tuple[bytes, bytes]):
        self.freq = freq
        self.key_int = key_int
        self.key_bytes = key_bytes
    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.key_bytes > other.key_bytes

class BPE:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.byte_word_list = [] # list[ByteWord]
        self.word_count = defaultdict(int) # dict[str : int]
        self.pair_hash = {} # dict[tuple[int, int] : PairValue]
        self.merge_list = [] # list[tuple[bytes, bytes]]
        self.vocab = {} # dict[int: bytes]

    def update_text(self, data: str):
        for text in data.split('<|endoftext|>'):
            word_list = re.findall(PAT, text)
            for w in word_list:
                self.word_count[w] += 1
    def update_word_count(self, word_count: dict[str : int]):
        for k,v in word_count.items():
            self.word_count[k] += v
    
    def train(self, num_merges: int):
        # remove special_tokens
        for t in self.special_tokens:
            self.word_count.pop(t, None)
        # update ByteWord
        self.byte_word_list = [ByteWord(k, v, idx) for idx, (k,v) in enumerate(self.word_count.items())]
        for byte_word in self.byte_word_list:
            byte_word.encode(self.pair_hash)
        for i in range(256):
            self.vocab[i] = bytes([i])
        # build heap
        heap = []
        for (k,v) in self.pair_hash.items():
            heapq.heappush(heap, QueueItem(v.freq, k, (self.vocab[k[0]], self.vocab[k[1]])))
        # merge
        while len(heap) > 0 and len(self.merge_list) < num_merges:
            queue_item = heapq.heappop(heap)
            freq = queue_item.freq
            key = queue_item.key_int
            if self.pair_hash[key].freq != freq:
                heapq.heappush(heap, QueueItem(self.pair_hash[key].freq, key, (self.vocab[key[0]], self.vocab[key[1]])))
                continue
            self.merge_list.append((self.vocab[key[0]], self.vocab[key[1]]))
            # print((self.vocab[key[0]], self.vocab[key[1]]), freq)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[key[0]] + self.vocab[key[1]]
            new_pairs = set()
            for idx in self.pair_hash[key].pos_idx:
                new_pairs = new_pairs | self.byte_word_list[idx].merge(key, new_id, self.pair_hash)
            for pair in new_pairs:
                heapq.heappush(heap, QueueItem(self.pair_hash[pair].freq, pair, (self.vocab[pair[0]], self.vocab[pair[1]])))
            if len(self.merge_list) % 1000 == 0:
                print(f'{len(self.merge_list)} / {num_merges}')
        for w in self.special_tokens:
            self.vocab[len(self.vocab)] = w.encode('utf-8')

def worker(data: str, queue: Queue):
    word_count = defaultdict(int) # dict[str : int]
    for text in data.split('<|endoftext|>'):
        word_list = re.findall(PAT, text)
        for w in word_list:
            word_count[w] += 1
    queue.put(word_count)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str])  \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] :
        ## Usage
    bpe = BPE(vocab_size, special_tokens)
    num_processes = 100
    with open(input_path, "rb") as f:
        print("Pretokenize")
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        print("BPE Prepare")
        q = Queue(maxsize=num_processes)
        processes = []
        results = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # bpe.update_text(chunk)
            p = Process(target=worker, args=(chunk, q))
            processes.append(p)
            p.start()
        for _ in processes:
            results.append(q.get())
            print(f'{len(results)} / {num_processes}')
        for p in processes:
            p.join()
        for word_count in results:
            bpe.update_word_count(word_count)
        num_merges = vocab_size - 256 - len(special_tokens)
        print("BPE Train")
        bpe.train(num_merges)
        return bpe.vocab, bpe.merge_list

if __name__ == "__main__":
    # vocab, merge_list = train_bpe('/root/assignment1-basics/tests/fixtures/corpus.en', 500, ["<|endoftext|>"])
    # vocab, merge_list = train_bpe('/root/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt', 1000, ["<|endoftext|>"])
    # vocab, merge_list = train_bpe('/root/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt', 10000, ["<|endoftext|>"])
    vocab, merge_list = train_bpe('/root/assignment1-basics/data/owt_train.txt', 32000, ["<|endoftext|>"])
    with open('merge.txt', 'w', encoding='utf-8') as f:
        for (a,b) in merge_list:
            f.write(f'{a} {b}\n')
    with open('merge.pkl', 'wb') as f:
        pickle.dump(merge_list, f)
    # vocab_str = {k: v.hex() for k,v in vocab.items()}
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for i in range(len(vocab)):
            f.write(f'{vocab[i]}\n')
        # json.dump(vocab_str, f, ensure_ascii=False, indent=2)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
