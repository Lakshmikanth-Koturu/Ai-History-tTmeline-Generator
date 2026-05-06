import os
import json
import urllib.request
import urllib.parse
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client_main = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client_backup = genai.Client(api_key=os.environ.get("BACKUP_GEMINI_API_KEY"))

def generate_with_retry(*args, **kwargs):
    max_retries = 4
    current_client = client_main
    for attempt in range(max_retries):
        try:
            return current_client.models.generate_content(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if '503' in error_msg or 'UNAVAILABLE' in error_msg or '429' in error_msg or 'quota' in error_msg.lower() or 'RESOURCE_EXHAUSTED' in error_msg:
                if attempt == 1:
                    print("Switching to backup API key...")
                    current_client = client_backup
                if attempt < max_retries - 1:
                    print(f"API busy (attempt {attempt + 1}/{max_retries}). Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
            raise

def fetch_wikipedia_context(topic):
    try:
        search_query = urllib.parse.quote(topic)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&utf8=&format=json&srlimit=1"
        user_agent = 'TimelineApp/1.0 (contact@example.com)'
        req = urllib.request.Request(search_url, headers={'User-Agent': user_agent})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            search_results = data.get('query', {}).get('search', [])
            if not search_results:
                return ""
            best_title = search_results[0]['title']
            
        summary_query = urllib.parse.quote(best_title)
        summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={summary_query}&format=json"
        req_sum = urllib.request.Request(summary_url, headers={'User-Agent': user_agent})
        with urllib.request.urlopen(req_sum, timeout=5) as response:
            sum_data = json.loads(response.read().decode('utf-8'))
            pages = sum_data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if 'extract' in page_info:
                    return f"EXTERNAL GROUNDING DATA (Wikipedia context for '{best_title}'):\n{page_info['extract'][:20000]}\n"
    except Exception as e:
        print("Wikipedia grounding error:", e)
    return ""

def get_wikipedia_image(search_term):
    try:
        search_query = urllib.parse.quote(search_term)
        # Using generator=search makes the query robust to slightly non-exact titles
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={search_query}&gsrlimit=1&prop=pageimages&pithumbsize=500&format=json"
        user_agent = 'TimelineApp/1.0 (contact@example.com)'
        req = urllib.request.Request(search_url, headers={'User-Agent': user_agent})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for page_id in pages:
                # Page IDs in generator search can be anything, check for thumbnail
                thumbnail = pages[page_id].get('thumbnail')
                if thumbnail and 'source' in thumbnail:
                    return thumbnail['source']
    except Exception as e:
        print(f"Error fetching image for {search_term}: {e}")
    return None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    history = data.get("history", [])
    new_message = data.get("message", "")
    
    history_str = ""
    for msg in history:
        role = "User" if msg.get('role') == 'user' else "Assistant"
        history_str += f"{role}: {msg.get('content')}\n"
        
    try:
        # Step 0: Extract the core subject for precise Wikipedia searching
        subject_response = generate_with_retry(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Extract the main historical topic or entity from this user message: '{new_message}'. If it is a greeting, general conversation, or short question with no historical entity, output 'NONE'. Otherwise output ONLY the topic name, nothing else.",
            config=types.GenerateContentConfig(temperature=0.1)
        )
        search_topic = subject_response.text.strip().strip('"').strip("'")
        
        wiki_context = ""
        if search_topic.upper() != "NONE":
            wiki_context = fetch_wikipedia_context(search_topic)
        prompt = (f"Conversation History:\n{history_str}\n\nNew User input: {new_message}\n\n{wiki_context}\n"
                  f"The current year is {datetime.datetime.now().year}. You are a historical timeline generator and a helpful AI assistant. "
                  f"If the user is greeting you, asking a general question, or engaging in small talk, respond concisely in 'chat_message' and leave 'enhanced_topic', 'timeline', and 'quiz' as null. "
                  f"ONLY generate a timeline if the user explicitly asks for one or provides a specific historical topic or entity. In those cases, generate a strictly fact-checked, highly comprehensive and detailed timeline using the provided External Grounding Data. You MUST extract as many key milestones as possible (aim for at least 10 to 20 detailed events spanning the entire history). "
                  f"Output ONLY a valid JSON object. Keys must be 'chat_message' (your conversational response to the user), "
                  f"'enhanced_topic' (string, if timeline generated), 'timeline' (list of objects, if timeline generated), "
                  f"and 'quiz' (list of 2-3 objects, if timeline generated, with 'question', 'options' (list of 4 strings), and 'correct_answer' keys). "
                  f"For a timeline, each object must have 'year' (string), 'description' (strictly accurate historical fact), 'detailed_info' (longer paragraph), "
                  f"and 'search_term' (highly specific 1-3 word exact Wikipedia page title of a famous person/place/artifact guaranteed to have a main image). "
                  f"For the quiz, create fun Multiple Choice Questions based on the timeline. Options must be plausible but only one correct. "
                  f"CRITICAL: DO NOT shift dates. Do not hallucinate. Output RAW JSON ONLY. No markdown formatting, no backticks, no text outside the JSON structure.")

        completion = generate_with_retry(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )
        
        response_content = completion.text.strip()
        if response_content.startswith("```"):
            lines = response_content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            response_content = "\n".join(lines).strip()
            
        temp_data = json.loads(response_content)

        if "timeline" in temp_data and temp_data["timeline"]:
            def fetch_image_for_event(event):
                if 'search_term' in event:
                    event['image_url'] = get_wikipedia_image(event['search_term'])
                return event
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                temp_data["timeline"] = list(executor.map(fetch_image_for_event, temp_data["timeline"]))

        return app.response_class(
            response=json.dumps(temp_data),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        print("Error:", e)
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype='application/json'
        )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
