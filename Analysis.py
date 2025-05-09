import os
import time
import json
import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Union
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential

# Third-party imports
import PyPDF2
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import openai
from openai.error import APIError, RateLimitError, Timeout
import pytesseract
from pdf2image import convert_from_path
import ftfy  # For text fixing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class TokenizationMethod(Enum):
    WORD = "word"
    SENTENCE = "sentence"
    SPACY = "spacy"
    REGEX = "regex"

@dataclass
class PreprocessingConfig:
    normalize_unicode: bool = True
    remove_stopwords: bool = True
    apply_lemmatization: bool = True
    tokenization_method: TokenizationMethod = TokenizationMethod.WORD
    preserve_numbers: bool = True
    preserve_punctuation: bool = True
    max_token_length: int = 100
    min_token_length: int = 2
    custom_stopwords: Set[str] = None
    
    def __post_init__(self):
        if self.custom_stopwords is None:
            self.custom_stopwords = set()

@dataclass
class DocumentMetadata:
    source_format: str
    total_pages: int = 0
    total_characters: int = 0
    total_tokens: int = 0
    encoding: str = "utf-8"
    language: str = "en"
    structure_info: Dict[str, any] = None
    
    def __post_init__(self):
        if self.structure_info is None:
            self.structure_info = {}

@dataclass
class Token:
    text: str
    position: int
    sentence_id: int
    page_number: int = 0
    is_stopword: bool = False
    lemma: str = ""
    pos_tag: str = ""
    category: str = "other"

def setup_openai_api() -> None:
    """Set up the OpenAI API key from environment variables."""
    api_key = os.environ.get('OPENAI_API_KEY')
    
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

def normalize_text(text: str, config: PreprocessingConfig) -> str:
    """
    Normalize text encoding and fix common issues.
    
    Args:
        text: Input text
        config: Preprocessing configuration
        
    Returns:
        Normalized text
    """
    if config.normalize_unicode:
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        # Normalize unicode (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters but keep newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t\r')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text

def extract_from_pdf(pdf_path: str, use_ocr: bool = False) -> Tuple[str, DocumentMetadata]:
    """
    Extract text from PDF with optional OCR support.
    
    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR for scanned pages
        
    Returns:
        Tuple of extracted text and metadata
    """
    text = ""
    metadata = DocumentMetadata(source_format="pdf")
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            metadata.total_pages = len(reader.pages)
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    
                    # If page is empty and OCR is enabled, try OCR
                    if not page_text.strip() and use_ocr:
                        logger.info(f"Using OCR for page {page_num + 1}")
                        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                        if images:
                            page_text = pytesseract.image_to_string(images[0])
                    
                    text += f"\n[PAGE {page_num + 1}]\n{page_text}"
                except Exception as e:
                    logger.error(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
                    
        metadata.total_characters = len(text)
        
    except Exception as e:
        logger.error(f"Error extracting from PDF: {str(e)}")
        raise
    
    return text, metadata

def extract_from_html(html_path: str) -> Tuple[str, DocumentMetadata]:
    """
    Extract text from HTML file.
    
    Args:
        html_path: Path to HTML file
        
    Returns:
        Tuple of extracted text and metadata
    """
    metadata = DocumentMetadata(source_format="html")
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Store structure information
        metadata.structure_info = {
            "title": soup.title.string if soup.title else "",
            "headings": [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3'])],
            "links": len(soup.find_all('a')),
            "images": len(soup.find_all('img'))
        }
        
        metadata.total_characters = len(text)
        
    except Exception as e:
        logger.error(f"Error extracting from HTML: {str(e)}")
        raise
    
    return text, metadata

def tokenize_text(text: str, config: PreprocessingConfig, metadata: DocumentMetadata) -> List[Token]:
    """
    Tokenize text according to configuration.
    
    Args:
        text: Input text
        config: Preprocessing configuration
        metadata: Document metadata
        
    Returns:
        List of Token objects
    """
    tokens = []
    
    # Split by pages if available
    pages = re.split(r'\[PAGE \d+\]', text)
    
    for page_num, page_text in enumerate(pages):
        if not page_text.strip():
            continue
            
        # Tokenize by sentences first
        sentences = sent_tokenize(page_text)
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Apply different tokenization methods
            if config.tokenization_method == TokenizationMethod.WORD:
                words = word_tokenize(sentence)
            elif config.tokenization_method == TokenizationMethod.SPACY and nlp:
                doc = nlp(sentence)
                words = [token.text for token in doc]
            elif config.tokenization_method == TokenizationMethod.REGEX:
                words = re.findall(r'\b\w+\b', sentence)
            else:
                # Fallback to simple split
                words = sentence.split()
            
            # Create token objects
            for pos, word in enumerate(words):
                # Filter by length
                if len(word) < config.min_token_length or len(word) > config.max_token_length:
                    continue
                
                # Create token
                token = Token(
                    text=word.lower(),
                    position=pos,
                    sentence_id=sent_idx,
                    page_number=page_num
                )
                
                # Check if stopword
                if config.remove_stopwords:
                    token.is_stopword = (
                        token.text in stopwords.words('english') or
                        token.text in ENGLISH_STOP_WORDS or
                        token.text in config.custom_stopwords
                    )
                
                # Apply lemmatization
                if config.apply_lemmatization and not token.text.isnumeric():
                    lemmatizer = WordNetLemmatizer()
                    token.lemma = lemmatizer.lemmatize(token.text)
                else:
                    token.lemma = token.text
                
                tokens.append(token)
    
    metadata.total_tokens = len(tokens)
    return tokens

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True
)
def classify_tokens_api(tokens: List[str]) -> Dict[str, List[str]]:
    """
    Classify tokens using OpenAI API with improved error handling.
    
    Args:
        tokens: List of tokens to classify
        
    Returns:
        Dictionary with classified tokens
    """
    example_block = """
Classify words into mathematical, software, machine_related, or other categories.

Examples:
mathematical: matrix, vector, derivative, entropy, algorithm, probability
software: programming, database, api, framework, debugging, interface
machine_related: neural, learning, gpu, cpu, processor, circuit
other: music, garden, novel, philosophy, culture, art

Return valid JSON with arrays: "mathematical", "software", "machine_related", "other".
"""
    
    prompt = example_block + "\nClassify these words:\n" + "\n".join(tokens)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You classify words into technical categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            timeout=30
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON if wrapped in markdown
            match = re.search(r'```(?:json)?\n?(.*?)```', content, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                # Fallback: all tokens to 'other'
                return {
                    "mathematical": [],
                    "software": [],
                    "machine_related": [],
                    "other": tokens
                }
        
        # Ensure all categories exist
        for category in ["mathematical", "software", "machine_related", "other"]:
            if category not in data:
                data[category] = []
        
        # Verify all tokens are classified
        classified_words = set()
        for words in data.values():
            classified_words.update(words)
        
        unclassified = [w for w in tokens if w not in classified_words]
        if unclassified:
            data["other"].extend(unclassified)
        
        return data
        
    except Exception as e:
        logger.error(f"API classification error: {str(e)}")
        # Fallback to rule-based classification
        return classify_tokens_rules(tokens)

def classify_tokens_rules(tokens: List[str]) -> Dict[str, List[str]]:
    """
    Fallback rule-based token classification.
    
    Args:
        tokens: List of tokens to classify
        
    Returns:
        Dictionary with classified tokens
    """
    mathematical_patterns = {
        'matrix', 'vector', 'derivative', 'integral', 'function', 'equation',
        'probability', 'statistics', 'algorithm', 'theorem', 'proof', 'lemma',
        'optimization', 'calculus', 'algebra', 'geometry', 'topology', 'metric'
    }
    
    software_patterns = {
        'code', 'program', 'software', 'development', 'programming', 'api',
        'database', 'framework', 'library', 'debug', 'compile', 'deploy',
        'version', 'git', 'repository', 'server', 'client', 'interface'
    }
    
    machine_patterns = {
        'machine', 'learning', 'neural', 'network', 'deep', 'model',
        'training', 'feature', 'dataset', 'classifier', 'regression',
        'accuracy', 'loss', 'gradient', 'backpropagation', 'epoch',
        'gpu', 'cpu', 'pytorch', 'tensorflow', 'sklearn'
    }
    
    results = {
        "mathematical": [],
        "software": [],
        "machine_related": [],
        "other": []
    }
    
    for token in tokens:
        token_lower = token.lower()
        
        # Check for partial matches
        if any(pattern in token_lower for pattern in mathematical_patterns):
            results["mathematical"].append(token)
        elif any(pattern in token_lower for pattern in software_patterns):
            results["software"].append(token)
        elif any(pattern in token_lower for pattern in machine_patterns):
            results["machine_related"].append(token)
        else:
            results["other"].append(token)
    
    return results

def categorize_tokens(tokens: List[Token], config: PreprocessingConfig) -> Dict[str, List[Token]]:
    """
    Categorize tokens into different categories.
    
    Args:
        tokens: List of Token objects
        config: Preprocessing configuration
        
    Returns:
        Dictionary with categorized tokens
    """
    # Separate tokens by type
    numeric_tokens = []
    stop_word_tokens = []
    to_classify_tokens = []
    
    for token in tokens:
        if token.text.isnumeric() or re.match(r'^[\d.,]+$', token.text):
            numeric_tokens.append(token)
            token.category = "numeric"
        elif token.is_stopword and config.remove_stopwords:
            stop_word_tokens.append(token)
            token.category = "stop_word"
        else:
            to_classify_tokens.append(token)
    
    # Classify non-numeric, non-stopword tokens
    if to_classify_tokens:
        unique_words = list(set(t.lemma for t in to_classify_tokens))
        
        try:
            # Try API classification first
            classification = classify_tokens_api(unique_words)
        except Exception as e:
            logger.warning(f"API classification failed: {str(e)}")
            # Fallback to rule-based
            classification = classify_tokens_rules(unique_words)
        
        # Map classifications back to tokens
        lemma_to_category = {}
        for category, words in classification.items():
            for word in words:
                lemma_to_category[word] = category
        
        # Update token categories
        for token in to_classify_tokens:
            category = lemma_to_category.get(token.lemma, "other")
            token.category = category
    
    # Group tokens by category
    categorized = {
        "numeric": numeric_tokens,
        "stop_words": stop_word_tokens,
        "mathematical": [t for t in to_classify_tokens if t.category == "mathematical"],
        "software": [t for t in to_classify_tokens if t.category == "software"],
        "machine_related": [t for t in to_classify_tokens if t.category == "machine_related"],
        "other": [t for t in to_classify_tokens if t.category == "other"]
    }
    
    return categorized

def process_document(
    file_path: str,
    config: PreprocessingConfig = None,
    use_ocr: bool = False
) -> Tuple[Dict[str, List[Token]], DocumentMetadata]:
    """
    Process a document with comprehensive preprocessing.
    
    Args:
        file_path: Path to the document
        config: Preprocessing configuration
        use_ocr: Whether to use OCR for PDFs
        
    Returns:
        Tuple of categorized tokens and metadata
    """
    if config is None:
        config = PreprocessingConfig()
    
    # Setup OpenAI API
    try:
        setup_openai_api()
    except ValueError as e:
        logger.warning(f"OpenAI API not available: {str(e)}")
    
    # Determine file type and extract text
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            text, metadata = extract_from_pdf(file_path, use_ocr)
        elif file_ext in ['.html', '.htm']:
            text, metadata = extract_from_html(file_path)
        else:
            # Try to read as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata = DocumentMetadata(source_format="text")
            metadata.total_characters = len(text)
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise
    
    # Normalize text
    with timer("Text normalization"):
        text = normalize_text(text, config)
    
    # Tokenize
    with timer("Tokenization"):
        tokens = tokenize_text(text, config, metadata)
    
    # Categorize tokens
    with timer("Token categorization"):
        categorized_tokens = categorize_tokens(tokens, config)
    
    # Generate summary statistics
    metadata.structure_info.update({
        "token_counts": {
            category: len(tokens) 
            for category, tokens in categorized_tokens.items()
        },
        "unique_words": {
            category: len(set(t.lemma for t in tokens))
            for category, tokens in categorized_tokens.items()
        }
    })
    
    return categorized_tokens, metadata

def export_results(
    categorized_tokens: Dict[str, List[Token]],
    metadata: DocumentMetadata,
    output_path: str = None
) -> None:
    """
    Export processing results to JSON.
    
    Args:
        categorized_tokens: Categorized tokens
        metadata: Document metadata
        output_path: Output file path
    """
    results = {
        "metadata": {
            "source_format": metadata.source_format,
            "total_pages": metadata.total_pages,
            "total_characters": metadata.total_characters,
            "total_tokens": metadata.total_tokens,
            "encoding": metadata.encoding,
            "language": metadata.language,
            "structure_info": metadata.structure_info
        },
        "tokens": {
            category: [
                {
                    "text": token.text,
                    "lemma": token.lemma,
                    "position": token.position,
                    "sentence_id": token.sentence_id,
                    "page_number": token.page_number,
                    "is_stopword": token.is_stopword,
                    "category": token.category
                }
                for token in tokens
            ]
            for category, tokens in categorized_tokens.items()
        }
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results exported to {output_path}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))

def main():
    """Main function with example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents with advanced text analysis')
    parser.add_argument('file_path', help='Path to the document to process')
    parser.add_argument('--output', '-o', help='Output file path for results')
    parser.add_argument('--ocr', action='store_true', help='Use OCR for scanned PDFs')
    parser.add_argument('--no-stopwords', action='store_true', help='Keep stop words')
    parser.add_argument('--no-lemma', action='store_true', help='Skip lemmatization')
    parser.add_argument('--tokenizer', choices=['word', 'sentence', 'spacy', 'regex'], 
                       default='word', help='Tokenization method')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PreprocessingConfig(
        remove_stopwords=not args.no_stopwords,
        apply_lemmatization=not args.no_lemma,
        tokenization_method=TokenizationMethod(args.tokenizer)
    )
    
    # Process document
    try:
        with timer("Total processing"):
            categorized_tokens, metadata = process_document(
                args.file_path,
                config=config,
                use_ocr=args.ocr
            )
        
        # Print summary
        print("\nPROCESSING SUMMARY")
        print("=" * 50)
        print(f"Source: {metadata.source_format}")
        print(f"Total tokens: {metadata.total_tokens}")
        
        for category, tokens in categorized_tokens.items():
            print(f"{category}: {len(tokens)} tokens")
        
        # Export results
        export_results(categorized_tokens, metadata, args.output)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
