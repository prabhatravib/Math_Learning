import os
import json
import logging
from typing import Dict, List
import openai
from openai.error import APIError, RateLimitError, Timeout
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

def setup_openai_api() -> None:
    """Set up the OpenAI API key from environment variables."""
    # Try to get from environment first
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # If not in environment, use the hardcoded key (only for development)
    if not api_key:
        api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual key when running
        logger.warning("Using hardcoded API key - for development only")
        
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        logger.warning("OPENAI_API_KEY not properly set")
        raise ValueError("API key not found or not properly set.")
        
    openai.api_key = api_key

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
