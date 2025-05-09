# Token Analysis and Visualization

This project analyzes PDF documents for different types of tokens (mathematical, software-related, machine learning-related, etc.) and creates visualizations comparing the token distributions across documents.

## Files Overview

1. **`Analysis_updated.py`** - Core analysis engine that extracts and classifies tokens from PDF files
2. **`token_visualization.py`** - Creates bar charts comparing token percentages across documents
3. **`run_token_analysis.py`** - Main script that orchestrates the analysis and visualization
4. **`README.md`** - This documentation file

## Requirements

### Python Libraries
```bash
pip install PyPDF2 scikit-learn openai matplotlib numpy tenacity
```

### OpenAI API Key
You need an OpenAI API key to classify tokens. Set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or modify the code to include your API key directly (not recommended for production).

## Quick Start

### 1. Basic Usage
```bash
python run_token_analysis.py --pdfs "path/to/your/document.pdf"
```

### 2. Analyze Multiple PDFs
```bash
python run_token_analysis.py --pdfs "doc1.pdf" "doc2.pdf" "doc3.pdf"
```

### 3. Analyze a Fraction of Each Document
```bash
python run_token_analysis.py --pdfs "large_document.pdf" --fraction 0.1
```

### 4. List Example PDFs
```bash
python run_token_analysis.py --list-example-pdfs
```

## Command Line Options

- `--pdfs` - List of PDF files to analyze
- `--fraction` - Fraction of each PDF to analyze (0.0-1.0, default: 1.0)
- `--output-chart` - Output filename for chart (default: token_comparison_chart.png)
- `--output-json` - Output filename for JSON results (default: token_analysis_results.json)
- `--load-existing` - Load existing JSON results instead of analyzing PDFs
- `--no-chart` - Skip creating the chart
- `--list-example-pdfs` - List example PDF paths

## Output Files

1. **JSON Results** - Detailed token counts for each document
   - Format: `token_analysis_results_YYYYMMDD_HHMMSS.json`

2. **Visualization Chart** - Bar chart comparing token percentages
   - Format: `token_comparison_chart_YYYYMMDD_HHMMSS.png`

## Token Categories

The analysis classifies tokens into the following categories:

1. **Mathematical** - Mathematical terms, equations, statistical concepts
2. **Software** - Programming languages, frameworks, development tools
3. **Machine-related** - AI/ML terms, hardware, neural networks
4. **Stop words** - Common words (the, and, of, etc.)
5. **Numbers and characters** - Numeric values and special characters

## Examples

### Example 1: Analyze Two ML Papers
```bash
python run_token_analysis.py --pdfs \
  "C:/Downloads/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf" \
  "C:/Downloads/A Few Useful Things to Know About Machine Learning.pdf"
```

### Example 2: Quick Analysis (10% of each document)
```bash
python run_token_analysis.py --pdfs "large_paper.pdf" --fraction 0.1
```

### Example 3: Load Existing Results and Recreate Chart
```bash
python run_token_analysis.py --load-existing --output-json "token_analysis_results_20240101_120000.json"
```

## Customization

### Modifying Token Classification

Edit the `classify_words_batch()` function in `Analysis_updated.py` to change how tokens are classified. The function uses OpenAI's GPT model to classify words based on examples.

### Customizing Visualization

Modify `create_token_comparison_chart()` in `token_visualization.py` to change:
- Chart colors
- Figure size
- Caption generation
- Additional metrics

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Ensure your API key is set correctly
   - Check your OpenAI account has sufficient credits
   - The script uses GPT-4 by default - ensure you have access

2. **PDF Reading Errors**
   - Some PDFs may be image-based and cannot be processed
   - Ensure PDFs are not password-protected

3. **Memory Issues**
   - For large PDFs, use the `--fraction` parameter to process a subset
   - The script processes tokens in batches to manage memory

### Debug Mode
Enable debug logging by modifying the logging level in any of the scripts:
```python
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is for educational purposes. Ensure you have the right to analyze the PDFs you process.

## Contact

For issues or questions, please check the logs for detailed error messages and troubleshooting information.
