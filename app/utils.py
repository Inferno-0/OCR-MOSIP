from thefuzz import fuzz

def verify_form_data(ocr_text: str, form_data: dict) -> dict:
    """
    Verify form data against OCR text using fuzzy matching.
    
    Args:
        ocr_text: Raw text extracted from OCR engine
        form_data: Dictionary of expected field values
                   Example: {"name": "John Doe", "invoice_id": "INV-2023-001"}
    
    Returns:
        {
            "total_score": int (0-100),
            "field_matches": {field_name: score, ...},
            "status": "verified" | "review_required"
        }
    
    Algorithm:
        1. For each field in form_data:
           - Use fuzzy partial_ratio to find best match in ocr_text
           - Score ranges from 0 (no match) to 100 (perfect match)
        2. Calculate aggregate total_score (average of all field scores)
        3. Determine status:
           - "verified" if total_score > 90
           - "review_required" if total_score <= 90
    """
    field_scores = {}
    
    # Iterate through each expected field
    for field_name, expected_value in form_data.items():
        # Convert to string in case of numeric values
        expected_str = str(expected_value)
        
        # Use partial_ratio: finds best matching substring
        # Example: "John Doe" in "Name: John Doe, Age: 30" = 100
        score = fuzz.partial_ratio(expected_str, ocr_text)
        
        field_scores[field_name] = score
    
    # Calculate aggregate score (average)
    if field_scores:
        total_score = sum(field_scores.values()) // len(field_scores)
    else:
        total_score = 0
    
    # Determine verification status
    status = "verified" if total_score > 90 else "review_required"
    
    return {
        "total_score": total_score,
        "field_matches": field_scores,
        "status": status
    }
