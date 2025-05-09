import json
import logging
from typing import Dict, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from api_client import classify_words_batch
from utils import timer

logger = logging.getLogger(__name__)

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

def classify_words(words: List[str], chunk_size: int = 200) -> Dict[str, List[str]]:
    """
    Classify words into 'mathematical', 'software', or 'other' categories.
    
    Args:
        words: List of words to classify
        chunk_size: Number of words to process in each batch
        
    Returns:
        Dictionary with "mathematical", "software", and "other" categories
    """
    results = {"mathematical": [], "software": [], "machine_related": [], "other": []}
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
                                    batch_results.get("machine_related", []) +
                                    batch_results.get("other", []))
                missing = set(batch) - returned_words
                if missing:
                    logger.warning(f"Tracking {len(missing)} unclassified words: {list(missing)[:5]}...")
                    unclassified_words.update(missing)
                
                results["mathematical"].extend(batch_results.get("mathematical", []))
                results["software"].extend(batch_results.get("software", []))
                results["machine_related"].extend(batch_results.get("machine_related", []))
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
    results["machine_related"] = list(set(results["machine_related"]))
    results["other"] = list(set(results["other"]))
    
    # Map results back to the original list
    word_to_category = {}
    for word in results["mathematical"]:
        word_to_category[word] = "mathematical"
    for word in results["software"]:
        word_to_category[word] = "software"
    for word in results["machine_related"]:
        word_to_category[word] = "machine_related"
    for word in results["other"]:
        word_to_category[word] = "other"
    
    # Check if any words are missing from the mapping
    missing_from_mapping = set(unique_words) - set(word_to_category.keys())
    if missing_from_mapping:
        logger.warning(f"Words missing from mapping: {missing_from_mapping}")
        for word in missing_from_mapping:
            word_to_category[word] = "other"
    
    final_results = {"mathematical": [], "software": [], "machine_related": [], "other": []}
    for word in words:
        if word in word_to_category:
            category = word_to_category[word]
            final_results[category].append(word)
        else:
            # This shouldn't happen, but just in case
            logger.warning(f"Word '{word}' not found in mapping, adding to 'other'")
            final_results["other"].append(word)
    
    logger.info(f"Classification complete: {len(final_results['mathematical'])} mathematical, {len(final_results['software'])} software, {len(final_results['machine_related'])} machine-related, {len(final_results['other'])} other")
    return final_results
