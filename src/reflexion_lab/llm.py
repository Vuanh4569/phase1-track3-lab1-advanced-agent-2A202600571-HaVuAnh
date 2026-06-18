import os
import time
import json
import urllib.request
from urllib.error import URLError, HTTPError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Select configuration
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Determine which API to use: "anthropic" or "openai"
if ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL:
    DEFAULT_PROVIDER = "anthropic"
    if not LLM_MODEL:
        LLM_MODEL = "claude-3-haiku-20240307"
elif GEMINI_API_KEY:
    DEFAULT_PROVIDER = "openai"
    OPENAI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
    OPENAI_API_KEY = GEMINI_API_KEY
    if not LLM_MODEL:
        LLM_MODEL = "gemini-1.5-flash"
else:
    DEFAULT_PROVIDER = "openai"
    if not LLM_MODEL:
        LLM_MODEL = "gpt-4o-mini"

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024, json_mode: bool = False) -> dict:
    """
    Call LLM using urllib.
    Returns a dict containing:
    - text: str
    - input_tokens: int
    - output_tokens: int
    - latency_ms: int
    """
    # Check mock mode
    if os.environ.get("MOCK_MODE", "false").lower() == "true":
        # Simulate local mock return
        time.sleep(0.05)
        return {
            "text": "MOCK_RESPONSE",
            "input_tokens": 100,
            "output_tokens": 50,
            "latency_ms": 50
        }

    provider = DEFAULT_PROVIDER
    # If base url contains openai, use openai format
    if "openai" in OPENAI_API_BASE.lower() or OPENAI_API_KEY:
        provider = "openai"

    start_time = time.time()
    
    if provider == "anthropic":
        url = ANTHROPIC_BASE_URL.rstrip('/') + '/v1/messages'
        headers = {
            'x-api-key': ANTHROPIC_AUTH_TOKEN,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        data = {
            'model': LLM_MODEL,
            'max_tokens': max_tokens,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_prompt}]
        }
        # Anthropic doesn't have standard json mode, but we can instruct in prompt
    else:
        # OpenAI or Gemini OpenAI format
        url = OPENAI_API_BASE.rstrip('/') + '/chat/completions'
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': LLM_MODEL,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': 0.0
        }
        if json_mode:
            data['response_format'] = {'type': 'json_object'}

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    # Retry logic up to 3 times
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                latency_ms = int((time.time() - start_time) * 1000)
                resp_data = json.loads(response.read().decode('utf-8'))
                
                if provider == "anthropic":
                    text = resp_data['content'][0]['text']
                    input_tokens = resp_data.get('usage', {}).get('input_tokens', 0)
                    output_tokens = resp_data.get('usage', {}).get('output_tokens', 0)
                else:
                    text = resp_data['choices'][0]['message']['content']
                    input_tokens = resp_data.get('usage', {}).get('prompt_tokens', 0)
                    output_tokens = resp_data.get('usage', {}).get('completion_tokens', 0)
                
                return {
                    "text": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms
                }
        except HTTPError as e:
            if attempt == 2:
                # print error response body
                error_body = e.read().decode('utf-8', errors='ignore')
                raise RuntimeError(f"LLM API call failed with {e.code}: {error_body}")
            time.sleep(1 * (attempt + 1))
        except (URLError, Exception) as e:
            if attempt == 2:
                raise RuntimeError(f"LLM API call failed: {str(e)}")
            time.sleep(1 * (attempt + 1))
            
    raise RuntimeError("LLM API call failed after max retries")
