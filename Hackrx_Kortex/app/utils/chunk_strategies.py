import gc
from typing import List, Dict, Any, Generator

def semantic_chunking(
    text_chunks: List[Dict[str, Any]], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Apply semantic chunking strategy to pre-processed document chunks.
    
    This function takes document sections/chunks and further splits them
    into semantic chunks with overlap to preserve context.
    
    Args:
        text_chunks: List of document chunks with text and metadata
        chunk_size: Maximum token/character size for each chunk
        chunk_overlap: Number of tokens/characters to overlap between chunks
        metadata: Additional metadata to add to each chunk
        
    Returns:
        List of semantic chunks with text and metadata
    """
    if not text_chunks:
        return []
        
    semantic_chunks = []
    global_metadata = metadata or {}
    
    # Set maximum text length to prevent memory issues
    MAX_TEXT_LENGTH = 100000  # Set a reasonable limit - 100K chars
    MAX_TOTAL_CHUNKS = 10000  # Limit total number of chunks to prevent memory issues
    
    for chunk_idx, chunk in enumerate(text_chunks):
        # Break early if we've hit the maximum number of chunks
        if len(semantic_chunks) >= MAX_TOTAL_CHUNKS:
            print(f"⚠️ Reached maximum chunk limit ({MAX_TOTAL_CHUNKS}), stopping chunking")
            break
            
        text = chunk["text"]
        chunk_metadata = chunk.get("metadata", {})
        
        # Combine metadata
        combined_metadata = {**global_metadata, **chunk_metadata}
        
        # Safety check - truncate excessively long text to prevent memory issues
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            print(f"⚠️ Warning: Truncated excessively long text in chunk {chunk_idx}")
        
        # If text is already smaller than chunk size, keep it as is
        if len(text) <= chunk_size:
            semantic_chunks.append({
                "text": text,
                "metadata": combined_metadata
            })
            continue
        
        # For very large texts, use a generator to avoid loading everything in memory
        for sub_chunk in _generate_semantic_chunks(
            text, 
            chunk_size, 
            chunk_overlap, 
            combined_metadata
        ):
            semantic_chunks.append(sub_chunk)
            
            # Break if we've hit the maximum number of chunks
            if len(semantic_chunks) >= MAX_TOTAL_CHUNKS:
                print(f"⚠️ Reached maximum chunk limit ({MAX_TOTAL_CHUNKS}), stopping chunking")
                break
            
        # Force garbage collection after processing every 10 chunks
        if chunk_idx % 10 == 0:
            gc.collect()
    
    return semantic_chunks
    
def _generate_semantic_chunks(
    text: str, 
    chunk_size: int, 
    chunk_overlap: int,
    metadata: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate semantic chunks from text without keeping all chunks in memory.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        metadata: Metadata to include in each chunk
        
    Yields:
        Chunks of text with metadata
    """
    current_position = 0
    chunk_index = 0
    
    while current_position < len(text):
        # Calculate end position for this chunk
        end_position = min(current_position + chunk_size, len(text))
        
        # If not at the end of the text, try to find a good break point
        if end_position < len(text):
            # Look for sentence boundaries - starting from end and moving backward
            # Stop at half the chunk size to ensure chunks aren't too small
            min_break_point = max(current_position + chunk_size // 2, current_position)
            
            # Fast search for common sentence endings
            found_break = False
            for i in range(end_position, min_break_point, -1):
                if i < len(text) and text[i] in ['.', '!', '?'] and (i + 1 >= len(text) or text[i + 1].isspace()):
                    end_position = i + 1
                    found_break = True
                    break
            
            # If no good break point found and chunk is large, force a break at a space
            if not found_break and end_position - current_position > chunk_size * 0.8:
                # Look for spaces as fallback
                for i in range(end_position, min_break_point, -1):
                    if i < len(text) and text[i].isspace():
                        end_position = i + 1
                        break
        
        # Create chunk with original metadata plus chunk index
        sub_chunk_metadata = {
            **metadata,
            "chunk_index": chunk_index
        }
        
        yield {
            "text": text[current_position:end_position],
            "metadata": sub_chunk_metadata
        }
        
        # Move position for next chunk, incorporating overlap
        current_position = end_position - chunk_overlap
        if current_position < 0:
            current_position = 0
            
        chunk_index += 1