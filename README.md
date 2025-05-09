# PDF Token Classification System

A modular system for extracting and classifying tokens from PDF documents using OpenAI's GPT-4.1 model, with automated validation capabilities.

## Project Structure

```
pdf_classification/
├── main.py                 # Main entry point and orchestration
├── pdf_processor.py        # PDF text extraction functionality
├── token_classifier.py     # Token categorization logic
├── api_client.py          # OpenAI API interaction
├── validation.py          # Validation and metrics calculation
├── utils.py               # Utility functions (timer)
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Features

- Extract tokens from PDF files
- Classify tokens into categories:
  - Mathematical terms
  - Software-related terms
  - Machine learning/AI/hardware terms
  - Stop words
  - Numeric characters
  - Other terms
- Automated validation with precision, recall, and F1 scores
- Random sampling for manual validation
- Modular architecture for easy maintenance

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from main import main

# Process a PDF and classify tokens
results = main("path/to/your/document.pdf", pdf_fraction=1.0)
```

### With Validation

```python
# Process PDF and run validation
results = main(
    "path/to/your/document.pdf", 
    pdf_fraction=1.0, 
    run_validation=True, 
    sample_size=500
)
```

### Module Usage

Each module can be used independently:

```python
# Extract tokens from PDF
from pdf_processor import extract_tokens_from_pdf
tokens = extract_tokens_from_pdf("document.pdf")

# Classify tokens
from token_classifier import classify_words
classified = classify_words(tokens)

# Run validation
from validation import validate_classification
validate_classification(results)
```

## Modules

### `pdf_processor.py`
- Extracts text tokens from PDF files
- Supports processing a fraction of the document
- Handles multi-page documents

### `token_classifier.py`
- Splits tokens into initial categories (numeric, stop words, to-classify)
- Uses OpenAI API to classify remaining tokens
- Handles batch processing for efficiency

### `api_client.py`
- Manages OpenAI API communication
- Implements retry logic for reliability
- Provides example-based classification prompts

### `validation.py`
- Selects random samples for manual validation
- Generates JSON files for manual labeling
- Calculates precision, recall, and F1 scores
- Provides formatted metric summaries

### `utils.py`
- Contains utility functions like the timer context manager

### `main.py`
- Orchestrates the entire pipeline
- Provides CLI interface
- Handles configuration and error management

## Validation Process

1. Random samples are selected from each category
2. A JSON file is generated for manual labeling
3. After manual labeling, metrics are automatically calculated
4. Results include precision, recall, and F1 scores for each category

## Configuration

Key parameters in `main.py`:
- `pdf_path`: Path to the PDF file
- `fraction`: Portion of PDF to process (0.0 to 1.0)
- `run_validation`: Whether to perform validation
- `sample_size`: Number of tokens to validate per category

## Error Handling

- Robust API error handling with retry logic
- Comprehensive logging throughout the system
- Graceful fallbacks for classification failures

## License

MIT License
