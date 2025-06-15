   # Sample 10 documents from TinyStories and OpenWebText and calculate compression ratios
import random
import numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

def sample_documents(file_path, num_samples=1000):
    # Only read first 100 lines
    documents = []
    current_doc = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Stop after 100 lines
                break
                
            if line.strip() == '' and current_doc:  # Empty line indicates end of document
                documents.append(''.join(current_doc))
                current_doc = []
            else:
                current_doc.append(line)
    
    # Handle the last document if it exists
    if current_doc:
        documents.append(''.join(current_doc))
    
    return random.sample(documents, min(num_samples, len(documents)))

def calculate_compression_ratio(text, tokenizer):
    # Get number of bytes in original text
    num_bytes = len(text.encode('utf-8'))
    # Get number of tokens after encoding
    num_tokens = len(tokenizer.encode(text))
    return num_bytes / num_tokens

def read_file_in_chunks(file_path, num_processes=10):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk

if __name__ == '__main__':

    # Question A
    print("\nQuestion A\n")
    # Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories 
    # and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), 
    # encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
    # Load tokenizers
    tiny_tokenizer = Tokenizer.from_files(
        vocab_filepath='bpe_result/tiny/vocab.pkl',
        merges_filepath='bpe_result/tiny/merge.pkl'
    )   

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath='bpe_result/owt/vocab.pkl',
        merges_filepath='bpe_result/owt/merge.pkl'
    )

    # Sample documents
    tiny_docs = sample_documents('data/TinyStoriesV2-GPT4-train.txt')
    owt_docs = sample_documents('data/owt_train.txt')

    # Calculate compression ratios
    tiny_ratios = [calculate_compression_ratio(doc, tiny_tokenizer) for doc in tiny_docs]
    owt_ratios = [calculate_compression_ratio(doc, owt_tokenizer) for doc in owt_docs]

    print(f"TinyStories tokenizer (10K vocab) average compression ratio: {sum(tiny_ratios)/len(tiny_ratios):.2f} bytes/token")
    print(f"OpenWebText tokenizer (32K vocab) average compression ratio: {sum(owt_ratios)/len(owt_ratios):.2f} bytes/token")

    # Question B
    print("Question B")
    # What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? 
    # Compare the compression ratio and/or qualitatively describe what happens.
    tiny_owt_ratios = [calculate_compression_ratio(doc, tiny_tokenizer) for doc in owt_docs]
    print(f"TinyStories tokenizer (10K vocab) average compression ratio for OpenWebText: {sum(tiny_owt_ratios)/len(tiny_owt_ratios):.2f} bytes/token")
    owt_tiny_ratios = [calculate_compression_ratio(doc, owt_tokenizer) for doc in tiny_docs]
    print(f"OpenWebText tokenizer (32K vocab) average compression ratio for TinyStories: {sum(owt_tiny_ratios)/len(owt_tiny_ratios):.2f} bytes/token")

    # Question C
    print("\nQuestion C\n")
    # Estimate the throughput of your tokenizer (e.g., in bytes/second). 
    # How long would it take to tokenize the Pile dataset (825GB of text)?
    # Measure tokenizer throughput
    import time

    def measure_tokenizer_throughput(tokenizer, text, num_runs=5):
        # Warm up
        for _ in range(2):
            tokenizer.encode(text)
        
        # Measure time
        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            tokenizer.encode(text)
            end_time = time.time()
            total_time += (end_time - start_time)
        
        avg_time = total_time / num_runs
        bytes_per_second = len(text.encode('utf-8')) / avg_time
        return bytes_per_second

    # Use a larger sample for more accurate measurement
    sample_size = 100000  # 100KB of text
    tiny_sample = ''.join(tiny_docs)[:sample_size]
    owt_sample = ''.join(owt_docs)[:sample_size]

    # Measure throughput for both tokenizers
    tiny_throughput = measure_tokenizer_throughput(tiny_tokenizer, tiny_sample)
    owt_throughput = measure_tokenizer_throughput(owt_tokenizer, owt_sample)

    print(f"TinyStories tokenizer throughput: {tiny_throughput/1e6:.2f} MB/s")
    print(f"OpenWebText tokenizer throughput: {owt_throughput/1e6:.2f} MB/s")

    # Calculate time to tokenize The Pile
    pile_size_bytes = 825 * 1024 * 1024 * 1024  # 825GB in bytes
    tiny_time = pile_size_bytes / tiny_throughput
    owt_time = pile_size_bytes / owt_throughput

    print(f"\nTime to tokenize The Pile (825GB):")
    print(f"Using TinyStories tokenizer: {tiny_time/3600:.2f} hours")
    print(f"Using OpenWebText tokenizer: {owt_time/3600:.2f} hours")

    # Question D
    # Using your TinyStories and OpenWebText tokenizers, 
    # encode the respective training and development datasets into a sequence of integer token IDs. 
    # We’ll use this later to train our language model. 
    # We recommend serializing the token IDs as a NumPy array of datatype uint16. 
    # Why is uint16 an appropriate choice?
    print("\nQuestion D\n")
    
    # Encode datasets using respective tokenizers
    tiny_train_ids = []
    tiny_valid_ids = []
    owt_train_ids = []
    owt_valid_ids = []
    
    # Process TinyStories datasets using multiprocessing
    num_processes = 32
    
    def tiny_encode_chunk(chunk):
        return tiny_tokenizer.encode(chunk)
    def owt_encode_chunk(chunk):
        return owt_tokenizer.encode(chunk)
    
    print("Encoding TinyStories train data...")
    with Pool() as pool:
        tiny_train_ids = []
        for chunk_ids in pool.imap(tiny_encode_chunk, read_file_in_chunks('data/TinyStoriesV2-GPT4-train.txt', num_processes)):
            tiny_train_ids.extend(chunk_ids)
    tiny_train_ids = np.array(tiny_train_ids, dtype=np.uint16)
    print(f"#Tokens: {len(tiny_train_ids)}")
    np.save('data/tiny_train_ids.npy', tiny_train_ids)
    del tiny_train_ids

    print("Encoding TinyStories valid data...")
    with Pool() as pool:
        tiny_valid_ids = []
        for chunk_ids in pool.imap(tiny_encode_chunk, read_file_in_chunks('data/TinyStoriesV2-GPT4-valid.txt', num_processes)):
            tiny_valid_ids.extend(chunk_ids)
    tiny_valid_ids = np.array(tiny_valid_ids, dtype=np.uint16)
    print(f"#Tokens: {len(tiny_valid_ids)}")
    np.save('data/tiny_valid_ids.npy', tiny_valid_ids)
    del tiny_valid_ids

    print("Encoding OpenWebText train data...")
    with Pool() as pool:
        owt_train_ids = []
        for chunk_ids in pool.imap(owt_encode_chunk, read_file_in_chunks('data/owt_train.txt', num_processes)):
            owt_train_ids.extend(chunk_ids)
    owt_train_ids = np.array(owt_train_ids, dtype=np.uint16)
    print(f"#Tokens: {len(owt_train_ids)}")
    np.save('data/owt_train_ids.npy', owt_train_ids)
    del owt_train_ids

    print("Encoding OpenWebText valid data...")
    with Pool() as pool:
        owt_valid_ids = []
        for chunk_ids in pool.imap(owt_encode_chunk, read_file_in_chunks('data/owt_valid.txt', num_processes)):
            owt_valid_ids.extend(chunk_ids)
    owt_valid_ids = np.array(owt_valid_ids, dtype=np.uint16)
    print(f"#Tokens: {len(owt_valid_ids)}")
    np.save('data/owt_valid_ids.npy', owt_valid_ids)
    del owt_valid_ids
    