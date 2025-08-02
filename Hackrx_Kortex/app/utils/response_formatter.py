def clean_response(response: str) -> str:
    """
    Clean and format LLM response for better readability
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Cleaned response text
    """
    # Remove common prefixes that LLMs might add
    prefixes_to_remove = [
        "Based on the provided context, ",
        "According to the document, ",
        "The document states that ",
        "From the information provided, ",
        "Based on the document context, "
    ]
    
    cleaned = response
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    
    # Remove answer prefix
    if cleaned.startswith("Answer: "):
        cleaned = cleaned[len("Answer: "):]
    
    # Make first letter uppercase if it's not
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # Ensure ending punctuation
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    
    return cleaned.strip()