import spacy
from typing import List, Optional
from app.internal.tokenizer import count_tokens


tokenizer = spacy.load("en_core_web_sm")
def tokenize_string(input_string) -> List[str]:
    tokens = [token.text for token in tokenizer(input_string)]
    return tokens

#returns list of chunks of input string
def chunk_string(input_string: str, chunk_size: Optional[int] = 20) -> list[str]:
    
    input_str_size = count_tokens(input_string)
    if input_str_size <= chunk_size:
        return [input_string]
    else:
        tokenized_string = tokenize_string(input_string)
        chunks: list[str]=[]   # list of chunks of input string
        current_chunk: list[str] = []    # current chunk of tokens

        for token in tokenized_string:
            if len(current_chunk) < chunk_size:
                current_chunk.append(token)
            else:
                current_string_chunk = " ".join(current_chunk)
                chunks.append(current_string_chunk)
                current_chunk = [token]  # start a new chunk with the current token

        # Add the last chunk if it's not empty
        if current_chunk:
            current_string_chunk = " ".join(current_chunk)
            chunks.append(current_string_chunk)

        return chunks




