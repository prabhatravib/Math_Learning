import json
import random
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def select_random_samples(categorized_results: Dict[str, List[str]], sample_size: int = 500) -> Dict[str, List[str]]:
    """
    Select random samples from each category for validation.
    
    Args:
        categorized_results: Dictionary with categorized tokens
        sample_size: Number of samples to select from each category
        
    Returns:
        Dictionary with random samples from each category
    """
    samples = {}
    
    for category, tokens in categorized_results.items():
        if category == "total_tokens":
            continue
            
        if len(tokens) <= sample_size:
            samples[category] = tokens.copy()
            logger.info(f"{category}: Using all {len(tokens)} tokens (less than {sample_size})")
        else:
            samples[category] = random.sample(tokens, sample_size)
            logger.info(f"{category}: Selected {sample_size} random samples from {len(tokens)} tokens")
    
    return samples

def save_validation_samples(samples: Dict[str, List[str]], filename: str = "validation_samples.json") -> None:
    """
    Save validation samples to a JSON file for manual labeling.
    
    Args:
        samples: Dictionary with token samples from each category
        filename: Output filename
    """
    # Prepare the format for manual labeling
    validation_data = {
        "instructions": """
        Please manually label each token below with the correct category:
        - mathematical: Math/statistics terms (vector, matrix, probability, etc.)
        - software: Programming/software terms (code, database, api, etc.)
        - machine_related: Machine learning/AI/hardware terms (neural, hardware, learning, etc.)
        - stop_word: Common English stop words (the, and, of, etc.)
        - numeric: Numbers and simple characters
        - other: All other terms
        
        For each token, assign the most appropriate category.
        """,
        "samples_for_labeling": {}
    }
    
    # Combine all samples into a single list for labeling
    all_samples = []
    sample_origins = {}
    
    for category, tokens in samples.items():
        for token in tokens:
            all_samples.append(token)
            sample_origins[token] = category
    
    # Shuffle the samples so labeler doesn't know the original categories
    random.shuffle(all_samples)
    
    # Create the labeling format
    for token in all_samples:
        validation_data["samples_for_labeling"][token] = {
            "original_category": sample_origins[token],
            "manual_label": "",  # To be filled by manual labeler
            "notes": ""  # Optional notes field
        }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(all_samples)} tokens for manual labeling to {filename}")
    logger.info(f"Please manually label the tokens and save the file.")

def load_validation_results(filename: str = "validation_samples.json") -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load manually labeled validation results.
    
    Args:
        filename: Input filename with manual labels
        
    Returns:
        Tuple of (predicted_labels, true_labels) dictionaries
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data["samples_for_labeling"]
        
        predicted_labels = {
            "mathematical": [],
            "software": [],
            "machine_related": [],
            "stop_word": [],
            "numeric": [],
            "other": []
        }
        
        true_labels = {
            "mathematical": [],
            "software": [],
            "machine_related": [],
            "stop_word": [],
            "numeric": [],
            "other": []
        }
        
        for token, label_info in samples.items():
            original_category = label_info["original_category"]
            manual_label = label_info.get("manual_label", "").strip()
            
            if not manual_label:
                logger.warning(f"Token '{token}' has no manual label, skipping")
                continue
            
            # Map the categories
            if original_category == "stop_words":
                original_category = "stop_word"
            elif original_category == "numbers and characters":
                original_category = "numeric"
            
            # Add to predicted labels (what the system classified)
            if original_category in predicted_labels:
                predicted_labels[original_category].append(token)
            
            # Add to true labels (what the human labeled)
            if manual_label in true_labels:
                true_labels[manual_label].append(token)
            else:
                logger.warning(f"Invalid manual label '{manual_label}' for token '{token}'")
        
        return predicted_labels, true_labels
        
    except Exception as e:
        logger.error(f"Error loading validation results: {str(e)}")
        raise

def compute_validation_metrics(predicted_labels: Dict[str, List[str]], true_labels: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and F1 score for each category.
    
    Args:
        predicted_labels: Dictionary of predicted labels by category
        true_labels: Dictionary of true (manual) labels by category
        
    Returns:
        Dictionary with metrics for each category
    """
    metrics = {}
    
    # Get all unique tokens
    all_tokens = set()
    for tokens in predicted_labels.values():
        all_tokens.update(tokens)
    for tokens in true_labels.values():
        all_tokens.update(tokens)
    
    # Create mapping from token to labels
    token_to_predicted = {}
    for category, tokens in predicted_labels.items():
        for token in tokens:
            token_to_predicted[token] = category
    
    token_to_true = {}
    for category, tokens in true_labels.items():
        for token in tokens:
            token_to_true[token] = category
    
    # Calculate metrics for each category
    for category in predicted_labels.keys():
        # True positives: tokens that are both predicted and truly in this category
        predicted_in_category = set(predicted_labels.get(category, []))
        true_in_category = set(true_labels.get(category, []))
        
        tp = len(predicted_in_category & true_in_category)
        
        # False positives: tokens predicted as this category but actually in another
        fp = len(predicted_in_category - true_in_category)
        
        # False negatives: tokens that are truly this category but predicted as another
        fn = len(true_in_category - predicted_in_category)
        
        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_predicted": len(predicted_in_category),
            "total_true": len(true_in_category)
        }
    
    # Calculate overall metrics (macro-average)
    avg_precision = sum(m["precision"] for m in metrics.values()) / len(metrics)
    avg_recall = sum(m["recall"] for m in metrics.values()) / len(metrics)
    avg_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
    
    metrics["macro_average"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }
    
    return metrics

def print_validation_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted summary of validation metrics.
    
    Args:
        metrics: Dictionary with metrics for each category
    """
    print("\nVALIDATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for category, scores in metrics.items():
        if category == "macro_average":
            continue
        precision = scores["precision"]
        recall = scores["recall"]
        f1 = scores["f1"]
        print(f"{category:<15} {precision:.3f}{'':<9} {recall:.3f}{'':<9} {f1:.3f}")
    
    print("-" * 50)
    macro = metrics["macro_average"]
    print(f"{'MACRO AVERAGE':<15} {macro['precision']:.3f}{'':<9} {macro['recall']:.3f}{'':<9} {macro['f1']:.3f}")
    print("=" * 50)
    
    # Additional details
    print("\nDETAILED METRICS")
    print("-" * 80)
    print(f"{'Category':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'Pred':<8} {'True':<8}")
    print("-" * 80)
    
    for category, scores in metrics.items():
        if category == "macro_average":
            continue
        tp = scores["true_positives"]
        fp = scores["false_positives"]
        fn = scores["false_negatives"]
        pred_total = scores["total_predicted"]
        true_total = scores["total_true"]
        print(f"{category:<15} {tp:<6} {fp:<6} {fn:<6} {pred_total:<8} {true_total:<8}")

def validate_classification(results: Dict[str, List[str]], sample_size: int = 500, validation_file: str = "validation_samples.json") -> None:
    """
    Perform validation of the classification results.
    
    Args:
        results: Categorized results from process_pdf
        sample_size: Number of samples to validate from each category
        validation_file: Filename for saving/loading validation data
    """
    print("\nSTARTING VALIDATION PROCESS")
    print("=" * 50)
    
    # Step 1: Select random samples
    print(f"\nStep 1: Selecting {sample_size} random samples from each category...")
    samples = select_random_samples(results, sample_size)
    
    # Step 2: Save samples for manual labeling
    print(f"\nStep 2: Saving samples to {validation_file} for manual labeling...")
    save_validation_samples(samples, validation_file)
    
    print(f"\nValidation samples saved to '{validation_file}'")
    print("NEXT STEPS:")
    print("1. Open the file and manually label each token in the 'manual_label' field")
    print("2. Save the file after completing manual labeling")
    print("3. Run this function again with the same validation_file to compute metrics")
    
    # Check if file already has manual labels
    response = input("\nDo you want to load and analyze existing manual labels? (y/n): ")
    if response.lower() == 'y':
        try:
            # Step 3: Load manually labeled results
            print(f"\nStep 3: Loading manually labeled results from {validation_file}...")
            predicted_labels, true_labels = load_validation_results(validation_file)
            
            # Step 4: Compute validation metrics
            print("\nStep 4: Computing validation metrics...")
            metrics = compute_validation_metrics(predicted_labels, true_labels)
            
            # Step 5: Print summary
            print_validation_summary(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation analysis: {str(e)}")
            print(f"\nError: {str(e)}")
            print("Please ensure the validation file has been properly labeled.")
    else:
        print("\nValidation process paused. Please complete manual labeling and run again.")
        return None
