# simple-assistant

A Streamlit-based conversational AI application that provides customer support for Thoughtful AI's services.

## Overview

This application serves as a customer support interface for Thoughtful AI, answering questions about the company's AI agents (EVA, CAM, and PHIL) and other general inquiries. It uses semantic search with embeddings to match user queries with predefined answers and falls back to OpenAI's GPT-4 for generating responses to novel questions.

## Features

- **Semantic Search**: Uses OpenAI embeddings to find the most relevant predefined answers
- **Streaming Responses**: Real-time response generation with a typing effect
- **Fallback Mechanism**: Keyword matching as a backup when embeddings are unavailable
- **Adjustable Similarity Threshold**: Control how strictly queries need to match predefined content

## Requirements

- Python 3.7+
- Poetry (for dependency management)
- Streamlit
- OpenAI Python Client
- NumPy
- python-dotenv

## Installation

1. Install dependencies with Poetry:
   ```
   poetry install
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Ensure you have a `predefined_data.json` file with your Q&A content.

## Usage

Run the Streamlit app with Poetry:
```
poetry run streamlit run app.py
```

## Structure

- `app.py`: Main application file containing the `ThoughtfulAISupport` class
- `predefined_data.json`: JSON file containing predefined questions and answers
- `.env`: Environment file for API keys

## Customization

- Modify the `predefined_data.json` file to add or update Q&A pairs
- Adjust the similarity threshold in the UI for stricter or looser matching
- Update the system prompt in the `generate_streaming_response` method to change the assistant's personality
