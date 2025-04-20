# Web Sources RAG

A Flask-based API service that combines YouTube video transcripts with web search results using advanced RAG (Retrieval-Augmented Generation) techniques for comprehensive content analysis.

## Features

- Extract and process YouTube video transcripts
- Perform Google Custom Search for relevant web content
- Multi-stage ranking system:
  - BM25 ranking for initial content relevance
  - LLM-based relevance scoring
  - Embedding-based semantic reranking
- Generate detailed responses using Google's Gemini 1.5 Flash model
- Containerized deployment with Docker

## Tech Stack

- Python 3.x
- Flask
- Google Gemini API
- Sentence Transformers
- YouTube Transcript API
- Docker

## Prerequisites

- Google Custom Search API Key
- Google Gemini API Key
- Custom Search Engine ID
- Docker (for containerized deployment)

## Environment Variables

Create a `.env` file with:
```
GOOGLE_CUSTOM_SEARCH_KEY=your_custom_search_key
CUSTOM_SEARCH_ENGINE_ID=your_search_engine_id
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MaleekNoob/web-sources-RAG.git
cd web-sources-RAG
```

2. Using Docker:
```bash
docker-compose up --build
```

Or manually:
```bash
pip install -r requirements.txt
python app.py
```

## API Usage

Send a POST request to `/analyze` endpoint:
```json
{
    "url": "youtube_video_url",
    "query": "your_analysis_query"
}
```

## Architecture

1. **Input Processing**:
   - YouTube video transcript extraction
   - Web content retrieval via Google Custom Search

2. **Ranking Pipeline**:
   - BM25 initial ranking
   - LLM-based relevance scoring
   - Semantic reranking using embeddings

3. **Response Generation**:
   - Context fusion of video and web content
   - Structured response generation using Gemini 1.5

## Contributing

Feel free to open issues and submit pull requests.

## License

This project is open source and available under the MIT License.