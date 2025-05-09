import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def timer(description: str):
    """Context manager to time operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} took {elapsed:.2f} seconds")
