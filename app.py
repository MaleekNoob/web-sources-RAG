import re
import requests
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from rank_bm25 import BM25Okapi
import googleapiclient.discovery
import googleapiclient.errors
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_KEY')
SEARCH_ENGINE_ID = os.getenv('CUSTOM_SEARCH_ENGINE_ID')
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Initialize models
model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Your existing functions (unchanged)
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(entry['text'] for entry in transcript)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        return f"Error: {e}"

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'q': query, 'key': GOOGLE_API_KEY, 'cx': SEARCH_ENGINE_ID, 'num': num_results}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        snippets = [item['snippet'] for item in results.get("items", [])]
        logging.info("Web search results fetched successfully.")
        return snippets
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching search results: {e}")
        return []

def rank_snippets(query, snippets):
    if not snippets:
        return []
    tokenized_corpus = [snippet.split() for snippet in snippets]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    ranked_snippets = sorted(zip(snippets, scores), key=lambda x: x[1], reverse=True)
    return [snippet for snippet, _ in ranked_snippets[:3]]

def rank_relevance(query, snippets):
    if not snippets:
        return []
    scored_snippets = []
    for snippet in snippets:
        prompt = f"On a scale of 1-10, how relevant is this snippet to '{query}'? Reply with ONLY the number:\n\n{snippet}"
        try:
            response = model.generate_content(prompt)
            score = int(re.search(r'\b([1-9]|10)\b', response.text.strip()).group(1))
            scored_snippets.append((snippet, score))
        except Exception as e:
            logging.error(f"Error in LLM-based ranking: {e}")
            scored_snippets.append((snippet, 0))
    return sorted(scored_snippets, key=lambda x: x[1], reverse=True)[:3]

def embedding_rerank(query, snippets):
    if not snippets:
        return []
    snippets = [snippet if isinstance(snippet, str) else snippet[0] for snippet in snippets]
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    snippet_embeddings = embedding_model.encode(snippets, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, snippet_embeddings)[0]
    ranked_snippets = sorted(zip(snippets, similarities), key=lambda x: x[1], reverse=True)
    return [snippet for snippet, _ in ranked_snippets[:3]]

def generate_response(video_transcript, web_snippets, query):
    combined_content = f"""
    Below is relevant content extracted from a YouTube video transcript and web sources.

    === Video Transcript (Extract) ===
    {video_transcript[:2000]}

    === Web Snippets ===
    {"\\n\\n".join(web_snippets)}  # Escape backslashes in the f-string

    === Task ===
    Provide a **detailed, structured response** to the query: "{query}". 
    - **Contextualize the information** and explain its significance.
    - **Compare different sources**, highlighting key points.
    - If applicable, **predict future trends or suggest actions**.
    - Ensure **clarity, coherence, and depth**.
    """
    response = model.generate_content(combined_content)
    return response.text.strip()

# API Endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    query = data.get('query')

    if not url or not query:
        return jsonify({'error': 'Missing URL or query'}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    video_transcript = get_transcript(video_id)
    web_results = google_search(query)
    bm25_ranked = rank_snippets(query, web_results)
    llm_filtered = rank_relevance(query, bm25_ranked)
    embedding_ranked = embedding_rerank(query, llm_filtered)
    final_response = generate_response(video_transcript, embedding_ranked, query)

    return jsonify({'response': final_response})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)