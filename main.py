import os
import sys
import logging
from typing import Dict, List
import openai

# Import modules
from api_client import setup_openai_api
from pdf_processor import extract_tokens_from_pdf
from token_classifier import split_tokens_by_type, classify_words
from validation import validate_classification, print_validation_summary
from utils import timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path: str, fraction: float = 0.1) -> Dict[str, List[str]]:
    """
    Process PDF and classify its content.
    
    Args:
        pdf_path: Path to the PDF file
        fraction: Fraction of the PDF to process (0.0 to 1.0)
        
    Returns:
        Dictionary with categorized tokens
    """
    # Setup OpenAI API
    try:
        setup_openai_api()
    except ValueError as e:
        logger.error(str(e))
        return {}
    
    try:
        # Extract tokens from PDF
        with timer("PDF extraction"):
            tokens = extract_tokens_from_pdf(pdf_path, fraction)
            
        logger.info(f"Total tokens from PDF: {len(tokens)}")
        
        # Count token occurrences for accurate accounting
        token_counts = {}
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
                
        # Get unique tokens
        unique_tokens = list(token_counts.keys())
        logger.info(f"Number of unique tokens: {len(unique_tokens)}")
        
        # Split tokens by type
        with timer("Token splitting"):
            token_categories = split_tokens_by_type(unique_tokens)
            
        # Log token counts for debugging
        logger.info(f"Numeric tokens: {len(token_categories['numeric'])}")
        logger.info(f"Stop words: {len(token_categories['stop_words'])}")
        logger.info(f"To classify: {len(token_categories['to_classify'])}")
        
        # Reduce batch size for word classification to avoid timeouts
        with timer("Word classification"):
            classification = classify_words(token_categories
