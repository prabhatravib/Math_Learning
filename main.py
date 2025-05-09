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
            classification = classify_words(token_categories["to_classify"], chunk_size=50)
            
        # Log classification results for debugging
        logger.info(f"Mathematical terms: {len(classification.get('mathematical', []))}")
        logger.info(f"Software terms: {len(classification.get('software', []))}")
        logger.info(f"Machine-related terms: {len(classification.get('machine_related', []))}")
        logger.info(f"Other terms: {len(classification.get('other', []))}")
        
        # Create a dictionary mapping tokens to their categories for easier lookup
        token_to_category = {}
        
        # First, assign numeric tokens
        for token in token_categories['numeric']:
            token_to_category[token] = 'numeric'
            
        # Then, assign stop words
        for token in token_categories['stop_words']:
            token_to_category[token] = 'stop_word'
            
        # Then, assign classified tokens (these take precedence over numeric and stop words)
        for token in classification.get('mathematical', []):
            token_to_category[token] = 'mathematical'
        for token in classification.get('software', []):
            token_to_category[token] = 'software'
        for token in classification.get('machine_related', []):
            token_to_category[token] = 'machine_related'
        for token in classification.get('other', []):
            token_to_category[token] = 'other'
        
        # Any token not yet categorized is assigned to 'other'
        for token in unique_tokens:
            if token not in token_to_category:
                token_to_category[token] = 'other'
                logger.warning(f"Uncategorized token found: {token}")
                
        # Check if all tokens have been categorized
        for token in unique_tokens:
            if token not in token_to_category:
                logger.error(f"Token not categorized: {token}")
                
        # Create the final categorized lists, accounting for duplicates in the original text
        mathematical_words = []
        software_words = []
        machine_related_words = []
        stop_words = []
        numbers_and_chars = []
        
        # Go through the original tokens to maintain original order and duplicates
        for token in tokens:
            category = token_to_category.get(token, 'other')
            if category == 'mathematical':
                mathematical_words.append(token)
            elif category == 'software':
                software_words.append(token)
            elif category == 'machine_related':
                machine_related_words.append(token)
            elif category == 'stop_word':
                stop_words.append(token)
            else:  # numeric, other, or uncategorized
                numbers_and_chars.append(token)
        
        # Final count check
        final_count = (
            len(mathematical_words) + 
            len(software_words) + 
            len(machine_related_words) +
            len(stop_words) + 
            len(numbers_and_chars)
        )
        
        logger.info(f"Final count breakdown - Math: {len(mathematical_words)}, Software: {len(software_words)}, Machine: {len(machine_related_words)}, Stop: {len(stop_words)}, Numbers/Chars: {len(numbers_and_chars)}")
        logger.info(f"Final total count: {final_count}, Original token count: {len(tokens)}")
        
        # Combine results with new structure
        return {
            "total_tokens": len(tokens),
            "stop_words": stop_words,
            "mathematical": mathematical_words,
            "software": software_words,
            "machine_related": machine_related_words,
            "numbers and characters": numbers_and_chars
        }
    except Exception as e:
        logger.error(f"Error in process_pdf: {str(e)}")
        return {
            "total_tokens": len(tokens) if 'tokens' in locals() else 0,
            "stop_words": stop_words if 'stop_words' in locals() else [],
            "mathematical": mathematical_words if 'mathematical_words' in locals() else [],
            "software": software_words if 'software_words' in locals() else [],
            "machine_related": machine_related_words if 'machine_related_words' in locals() else [],
            "numbers and characters": numbers_and_chars if 'numbers_and_chars' in locals() else []
        }

def main(pdf_path: str, pdf_fraction: float = 0.1, run_validation: bool = False, sample_size: int = 500) -> Dict[str, List[str]]:
    """
    Main function to process a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_fraction: Fraction of the PDF to process (0.0 to 1.0)
        run_validation: Whether to run validation after classification
        sample_size: Number of samples for validation
        
    Returns:
        Dictionary with categorized tokens
    """
    try:
        # Test OpenAI API key first
        try:
            setup_openai_api()
            # Simple test to see if API key works
            logger.info("Testing OpenAI API connection...")
            test_response = openai.ChatCompletion.create(
                model="gpt-4.1",
                messages=[{"role": "system", "content": "Test connection"}],
                max_tokens=5
            )
            logger.info("OpenAI API connection successful.")
        except Exception as e:
            logger.error(f"OpenAI API test failed: {str(e)}")
            logger.error("Please check your API key and connection.")
            return {
                "error": f"OpenAI API test failed: {str(e)}",
                "total_tokens": 0,
                "numeric": [],
                "stop_words": [],
                "mathematical": [],
                "software": [],
                "machine_related": [],
                "other": []
            }
            
        # Process the PDF
        results = process_pdf(pdf_path, pdf_fraction)
        
        # If validation requested, run validation process
        if run_validation:
            print("\nInitiating validation process...")
            validation_metrics = validate_classification(results, sample_size)
            
            # Add validation metrics to results
            if validation_metrics:
                results["validation_metrics"] = validation_metrics
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            "error": str(e),
            "total_tokens": 0,
            "numeric": [],
            "stop_words": [],
            "mathematical": [],
            "software": [],
            "machine_related": [],
            "other": []
        }

if __name__ == "__main__":
    # Set the PDF path directly in the code
    pdf_path = r"C:\Users\prabh\Downloads\A Few Useful Things to Know About Machine Learning.pdf"
    # pdf_path = r"C:\Users\prabh\Downloads\Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"

    # Process all tokens from the PDF (set fraction to 1.0 for 100%)
    fraction = 1.0  # Process 100% of the PDF
    
    # Set run_validation to True to perform validation
    run_validation = True
    sample_size = 500
    
    results = main(pdf_path, fraction, run_validation, sample_size)
    
    # Print summary
    print(f"\nTOKEN CLASSIFICATION SUMMARY:")
    print(f"------------------------------")
    print(f"Total tokens processed: {results.get('total_tokens', 0)} ({fraction*100:.1f}% of PDF)")
    print(f"Stop words: {len(results.get('stop_words', []))}")
    print(f"Mathematical terms: {len(results.get('mathematical', []))}")
    print(f"Software terms: {len(results.get('software', []))}")
    print(f"Machine-related terms: {len(results.get('machine_related', []))}")
    print(f"Numbers and characters: {len(results.get('numbers and characters', []))}")
    
    # Calculate token coverage
    total_categorized = (
        len(results.get('stop_words', [])) + 
        len(results.get('mathematical', [])) + 
        len(results.get('software', [])) + 
        len(results.get('machine_related', [])) +
        len(results.get('numbers and characters', []))
    )
    
    # Sum of categories
    print(f"\nSum of categories: {total_categorized}")
    
    # Calculate coverage percentage
    coverage_percentage = (total_categorized / results.get('total_tokens', 1)) * 100
    print(f"Token coverage: {coverage_percentage:.2f}%")
    
    if coverage_percentage != 100.0:
        print(f"WARNING: Token coverage is not 100%. Some tokens may be missing or duplicated.")
    else:
        print(f"SUCCESS: All tokens accounted for.")
    
    # Optional: Print a sample of each category
    print("\nSAMPLE TERMS FROM EACH CATEGORY:")
    print(f"------------------------------")
    print("\nSample mathematical terms:", results.get('mathematical', [])[:10])
    print("\nSample software terms:", results.get('software', [])[:10])
    print("\nSample machine-related terms:", results.get('machine_related', [])[:10])
    print("\nSample numbers and characters:", results.get('numbers and characters', [])[:10])
    
    # Print validation results if available
    if "validation_metrics" in results:
        print("\nVALIDATION METRICS SUMMARY:")
        print_validation_summary(results["validation_metrics"])
