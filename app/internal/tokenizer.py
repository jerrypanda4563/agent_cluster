import tiktoken


tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    num_tokens = len(tokenizer.encode(text))
    return num_tokens