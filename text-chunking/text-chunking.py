def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    if len(tokens) == 0:
        return []
    chunk_size = min(chunk_size, len(tokens))
    overlap = min(overlap, chunk_size - 1)
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens) - chunk_size + 1, chunk_size - overlap)]