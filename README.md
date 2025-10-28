# BLS Job Description Generator

This is an experiment in using LLMs to write at scale.

For every occupation, the BLS provides dry descriptions of every job. We will attempt to take those descriptions and use them as grounding to generate copy at scale.

## Files

- **`text_gen.py`**: Main script that processes BLS data and generates more engaging job descriptions using Claude Sonnet 4. Features batch processing, auto-resume from partial results, and retry logic.

- **`model_testing.py`**: Benchmarking script that tests multiple AI models (Claude, GPT, Gemini, DeepSeek, etc.) at different temperature settings to compare output quality and characteristics. In my testing, all models performed similarly at this task, but Claude marginally better than the rest.

- **`test_two_models.py`**: Simplified testing script for comparing two specific models (DeepSeek and Gemini) with results appended to existing comparison data.

- **`bls_occupations_input.csv`**: Input data containing 877 BLS occupation titles and descriptions.

- **`bls_with_descriptions.csv`**: Final output with all 877 AI-generated job descriptions.

- **`bls_with_descriptions.partial.csv`**: Incremental backup saved after each batch.

- **`model_comparison.csv`**: Comparison results from model testing across 9 different LLMs.