import re
import logging
from typing import List
import PyPDF2

logger = logging.getLogger(__name__)

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
