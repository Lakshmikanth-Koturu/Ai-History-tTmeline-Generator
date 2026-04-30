import os
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# The API key is automatically loaded from the .env file
client = Groq()

def get_wikipedia_image(search_term):
    try:
        # First, search for the most relevant Wikipedia page title
        search_query = urllib.parse.quote(search_term)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&utf8=&format=json&srlimit=1"
        req_search = urllib.request.Request(search_url, headers={'User-Agent': 'TimelineApp/1.0'})
        with urllib.request.urlopen(req_search, timeout=3) as response:
            search_data = json.loads(response.read().decode('utf-8'))
            if search_data.get('query', {}).get('search'):
                best_title = search_data['query']['search'][0]['title']
                
                # Now fetch the summary for the exact best title
                title_quote = urllib.parse.quote(best_title)
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_quote}"
                req_summary = urllib.request.Request(summary_url, headers={'User-Agent': 'TimelineApp/1.0'})
                with urllib.request.urlopen(req_summary, timeout=3) as summary_response:
                    summary_data = json.loads(summary_response.read().decode('utf-8'))
                    if 'thumbnail' in summary_data and 'source' in summary_data['thumbnail']:
                        return summary_data['thumbnail']['source']
    except Exception as e:
        print(f"Error fetching image for {search_term}: {e}")
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    timeline_data = None
    topic = None
    error = None

    if request.method == "POST":
        topic = request.form.get("topic")
        if topic:
            try:
                # Call Groq API
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a historical timeline generator. First, analyze the user's topic. Correct any spelling mistakes, and enhance it into a clear, specific historical subject. Then, generate the timeline. Output ONLY a valid JSON object with the keys 'enhanced_topic' (string) and 'timeline' (list of objects). Each timeline object must have 'year' (string), 'description' (string), 'detailed_info' (string, a longer paragraph providing more in-depth context and facts about the event for a popup), and 'search_term' (string, a highly specific 1-3 word exact Wikipedia page title that represents the event for finding a relevant image). Include recent events up to the present day (2026). Do not include markdown code blocks or any other text."
                        },
                        {
                            "role": "user",
                            "content": f"Generate a highly detailed and continuous timeline of 15 to 20 key events for the topic: {topic}, ensuring you do not skip major periods or years and it includes the most up-to-date events up to the current year. Output in JSON format."
                        }
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                response_content = completion.choices[0].message.content
                print("Raw Response:", response_content)
                
                data = json.loads(response_content)
                if isinstance(data, dict) and "timeline" in data:
                    timeline_data = data["timeline"]
                    
                    if "enhanced_topic" in data:
                        topic = data["enhanced_topic"]
                    
                    # Fetch images concurrently
                    def fetch_image_for_event(event):
                        if 'search_term' in event:
                            event['image_url'] = get_wikipedia_image(event['search_term'])
                        return event
                    
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        timeline_data = list(executor.map(fetch_image_for_event, timeline_data))
                else:
                    error = "Unexpected response format from AI."
            except Exception as e:
                error = str(e)
                print("Error:", e)

    return render_template("index.html", timeline_data=timeline_data, topic=topic, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
