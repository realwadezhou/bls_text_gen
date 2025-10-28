# BLS Job Description Generator

This is an experiment in using LLMs to write at scale.

For every occupation, the BLS provides dry descriptions of every job. We will attempt to take those descriptions and use them as grounding to generate copy at scale.

## Files

- **`text_gen.py`**: Main script that processes BLS data and generates more engaging job descriptions using Claude Sonnet 4. Features batch processing, auto-resume from partial results, and retry logic.

- **`model_testing.py`**: Benchmarking script that tests multiple AI models (Claude, GPT, Gemini, DeepSeek, etc.) at different temperature settings to compare output quality and characteristics. In my testing, all models performed similarly at this task, but Claude felt the best.

- **`test_two_models.py`**: Simplified testing script for comparing two specific models (DeepSeek and Gemini) with results appended to existing comparison data.

- **`bls_occupations_input.csv`**: Input data containing 877 BLS occupation titles and descriptions.

- **`bls_with_descriptions.csv`**: Final output with all generated descriptions (gitignored).

- **`bls_with_descriptions.partial.csv`**: Incremental backup saved after each batch (gitignored).

- **`model_comparison.csv`**: Comparison results from model testing scripts (gitignored).

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenRouter API key:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

Generate descriptions:
```bash
python text_gen.py
```

Test different models:
```bash
python model_testing.py
```

## Features

- **Batch processing**: Reduces API calls and costs
- **Auto-resume**: Continues from partial results if interrupted  
- **Multiple model support**: Test and compare different LLMs
- **Robust error handling**: Retries with exponential backoff
- **Quality examples**: Few-shot prompting with curated examples

## Requirements

- Python 3.7+
- OpenRouter API key
- Internet connection for API calls

