# Models Classify README

This directory contains Python files for classifying data using various AI models. To use these files, you need to create a `.env` file in the root of your project to store API keys for the models you want to use.

## Required API Keys

The `.env` file should include the following keys:

- **OPENAI_API_KEY**: API key for OpenAI's models.
- **GOOGLE_API_KEY**: API key for Google's Gemini model.
- **ANTHROPIC_API_KEY**: API key for Anthropic's Claude model.

### Example `.env` File

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Make sure to replace `your_openai_api_key_here`, `your_google_api_key_here`, and `your_anthropic_api_key_here` with your actual API keys.

## Usage

Once the `.env` file is set up, you can run the Python files in this directory to interact with the models. Ensure you have installed all required dependencies and have access to the APIs. The .gitignore file automatically ignores your .env file, so you can still use git and not be worried about sharing your api-keys.
