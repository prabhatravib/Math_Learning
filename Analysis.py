import os
import time
import json
import re
from typing import Dict, List, Set, Tuple, Optional
import logging
from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_exponential

# Third-party imports
import PyPDF2
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import openai
from openai.error import APIError, RateLimitError, Timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up OpenAI API key
def setup_openai_api() -> None:
    """Set up the OpenAI API key from environment variables."""
    # Try to get from environment first
    api_key = OPENAI_API_KEY
    
    # If not in environment, use the hardcoded key (only for development)
    if not api_key:
        api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual key when running
        logger.warning("Using hardcoded API key - for development only")
        
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        logger.warning("OPENAI_API_KEY not properly set")
        raise ValueError("API key not found or not properly set.")
        
    openai.api_key = api_key

@contextmanager
def timer(description: str):
    """Context manager to time operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} took {elapsed:.2f} seconds")

def split_tokens_by_type(tokens: List[str]) -> Dict[str, List[str]]:
    """
    Split tokens into numeric, stop words, and to-classify categories.
    
    Args:
        tokens: List of tokens extracted from text
        
    Returns:
        Dictionary with categorized tokens
    """
    results = {
        "numeric": [],
        "stop_words": [],
        "to_classify": []
    }
    
    for token in tokens:
        # Try to convert to float to check if it's numeric
        try:
            float(token)
            results["numeric"].append(token)
        except ValueError:
            # If not numeric, check if it's a stop word
            if token in ENGLISH_STOP_WORDS:
                results["stop_words"].append(token)
            else:
                # If not numeric or stop word, it needs classification
                results["to_classify"].append(token)
    
    return results

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True
)
def classify_words_batch(words: List[str]) -> Dict[str, List[str]]:
    """
    Classify a batch of words into 'mathematical', 'software', 'machine_related', or 'other' categories.
    
    Args:
        words: List of words to classify
        
    Returns:
        Dictionary with "mathematical", "software", "machine_related", and "other" categories
    """
    example_block = """
Here are some examples:
  statistics    -> mathematical
  matrix        -> mathematical
  vector        -> mathematical
  function      -> mathematical
  probability   -> mathematical
  derivative    -> mathematical
  algorithm     -> mathematical
  entropy       -> mathematical
  calculus      -> mathematical
  geometry      -> mathematical
  programming   -> software
  database      -> software
  code          -> software
  interfaces    -> software
  compiler      -> software
  debugging     -> software
  framework     -> software
  developer     -> software
  api           -> software
  architecture  -> software
  machine       -> machine_related
  learning      -> machine_related
  hardware      -> machine_related
  circuit       -> machine_related
  processor     -> machine_related
  memory        -> machine_related
  cpu           -> machine_related
  gpu           -> machine_related
  device        -> machine_related
  neural        -> machine_related
  music         -> other
  language      -> other
  history       -> other
  garden        -> other
  novel         -> other
  philosophy    -> other
  happiness     -> other
  culture       -> other
  art           -> other
  biology       -> other

Now classify the following words. Return only valid JSON with four arrays: "mathematical", "software", "machine_related", and "other".
"""
    prompt = example_block + "\nWords:\n" + "\n".join(words)
    
    try:
        logger.info(f"Sending batch of {len(words)} words to OpenAI API")
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You classify words or phrases into mathematical, software, machine_related (including machine learning, AI, neural networks, hardware), and other categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            timeout=30
        )
        
        content = response.choices[0].message.content
        logger.info(f"Received response from OpenAI API (length: {len(content)} chars)")
        
        if not content:
            logger.warning("Empty response from API")
            return {"mathematical": [], "software": [], "machine_related": [], "other": words}
        
        # Clean the response to ensure it's valid JSON
        content = content.strip()
        # If the response isn't already JSON (wrapped in {}), add the braces
        if not (content.startswith('{') and content.endswith('}')):
            if '{"mathematical":' in content or '"software":' in content or '"machine_related":' in content or '"other":' in content:
                # Extract just the JSON part if possible
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    content = content[start_idx:end_idx]
                else:
                    # Fallback: wrap the response in braces to try to make it valid JSON
                    content = '{' + content + '}'
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response content: {content}")
            # Default categorization if JSON parsing fails
            return {"mathematical": [], "software": [], "machine_related": [], "other": words}
        
        # Make sure all categories exist
        if "mathematical" not in data:
            data["mathematical"] = []
        if "software" not in data:
            data["software"] = []
        if "machine_related" not in data:
            data["machine_related"] = []
        if "other" not in data:
            data["other"] = []
            
        # Find any words that aren't classified and add them to "other"
        classified_words = set(data["mathematical"] + data["software"] + data["machine_related"] + data["other"])
        unclassified_words = [w for w in words if w not in classified_words]
        
        if unclassified_words:
            # Simply add any unclassified words to "other" category
            data["other"].extend(unclassified_words)
            logger.info(f"Added {len(unclassified_words)} unclassified words to 'other' category")
            
        return data
        
    except Exception as e:
        logger.error(f"Error classifying words: {str(e)}")
        raise

def classify_words(words: List[str], chunk_size: int = 200) -> Dict[str, List[str]]:
    """
    Classify words into 'mathematical', 'software', or 'other' categories.
    
    Args:
        words: List of words to classify
        chunk_size: Number of words to process in each batch
        
    Returns:
        Dictionary with "mathematical", "software", and "other" categories
    """
    results = {"mathematical": [], "software": [], "other": []}
    unclassified_words = set()
    
    # Reduce data size by using only unique words
    unique_words = list(set(words))
    logger.info(f"Classifying {len(unique_words)} unique words out of {len(words)} total")
    
    # Use smaller batches for more reliable processing
    chunk_size = min(chunk_size, 50)  # Limit batch size to prevent timeouts
    
    # For testing, let's use a simpler approach if we run into issues
    failure_count = 0
    max_failures = 3
    
    # Process in batches
    for i in range(0, len(unique_words), chunk_size):
        with timer(f"Batch classification {i//chunk_size + 1}"):
            batch = unique_words[i:i + chunk_size]
            try:
                batch_results = classify_words_batch(batch)
                
                # Track any words that weren't classified
                returned_words = set(batch_results.get("mathematical", []) + 
                                    batch_results.get("software", []) + 
                                    batch_results.get("other", []))
                missing = set(batch) - returned_words
                if missing:
                    logger.warning(f"Tracking {len(missing)} unclassified words: {list(missing)[:5]}...")
                    unclassified_words.update(missing)
                
                results["mathematical"].extend(batch_results.get("mathematical", []))
                results["software"].extend(batch_results.get("software", []))
                results["other"].extend(batch_results.get("other", []))
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                failure_count += 1
                # Put all words in "other" category for this batch
                results["other"].extend(batch)
                
                # If we've had too many failures, switch to a simpler approach
                if failure_count >= max_failures:
                    logger.warning("Too many failures, switching to simple classification")
                    remaining_words = unique_words[i+chunk_size:]
                    # Simple classification - put all remaining words in "other"
                    results["other"].extend(remaining_words)
                    break
    
    # Final check for unclassified words at the end
    if unclassified_words:
        logger.warning(f"Final check found {len(unclassified_words)} unclassified words")
        results["other"].extend(unclassified_words)
        logger.info(f"Added all unclassified words to 'other' category")
    
    # Deduplicate results
    results["mathematical"] = list(set(results["mathematical"]))
    results["software"] = list(set(results["software"]))
    results["other"] = list(set(results["other"]))
    
    # Map results back to the original list
    word_to_category = {}
    for word in results["mathematical"]:
        word_to_category[word] = "mathematical"
    for word in results["software"]:
        word_to_category[word] = "software"
    for word in results["other"]:
        word_to_category[word] = "other"
    
    # Check if any words are missing from the mapping
    missing_from_mapping = set(unique_words) - set(word_to_category.keys())
    if missing_from_mapping:
        logger.warning(f"Words missing from mapping: {missing_from_mapping}")
        for word in missing_from_mapping:
            word_to_category[word] = "other"
    
    final_results = {"mathematical": [], "software": [], "other": []}
    for word in words:
        if word in word_to_category:
            category = word_to_category[word]
            final_results[category].append(word)
        else:
            # This shouldn't happen, but just in case
            logger.warning(f"Word '{word}' not found in mapping, adding to 'other'")
            final_results["other"].append(word)
    
    logger.info(f"Classification complete: {len(final_results['mathematical'])} mathematical, {len(final_results['software'])} software, {len(final_results['other'])} other")
    return final_results

def extract_tokens_from_pdf(pdf_path: str, fraction: float = 1.0) -> List[str]:
    """
    Extract tokens from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        fraction: Fraction of the PDF to process (0.0 to 1.0)
        
    Returns:
        List of tokens extracted from the PDF
    """
    all_tokens = []
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            
            logger.info(f"PDF has {total_pages} pages")
            
            # First, count total tokens to determine sample size
            for page in reader.pages:
                text = page.extract_text() or ""
                page_tokens = re.findall(r"\b\w+\b", text.lower())
                all_tokens.extend(page_tokens)
            
            total_tokens = len(all_tokens)
            logger.info(f"Total tokens in PDF: {total_tokens}")
            
            # Calculate number of tokens to process based on fraction
            tokens_to_process = int(total_tokens * fraction)
            logger.info(f"Processing {tokens_to_process} tokens ({fraction * 100:.1f}% of total)")
            
            # Return the subset of tokens
            return all_tokens[:tokens_to_process]
                    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

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

def main(pdf_path: str, max_words: int = 10000) -> Dict[str, List[str]]:
    """
    Main function to process a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        max_words: Maximum number of words to process
        
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
                "other": []
            }
            
        return process_pdf(pdf_path, max_words)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            "error": str(e),
            "total_tokens": 0,
            "numeric": [],
            "stop_words": [],
            "mathematical": [],
            "other": []
        }

    import sys
    
    # Set the PDF path directly in the code
    pdf_path=r"C:\Users\prabh\Downloads\A Few Useful Things to Know About Machine Learning.pdf"
    pdf_path = r"C:\Users\prabh\Downloads\Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"

    # Process all tokens from the PDF (set fraction to 1.0 for 100%)
    fraction = 1.0  # Process 100% of the PDF
    results = main(pdf_path, fraction)
    
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
