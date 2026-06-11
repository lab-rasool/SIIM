"""
Tiny Ollama HTTP client shared by the SIIM Local LLMs lab scripts.

Why a thin wrapper instead of a heavy SDK?
- The whole point of the workshop is "the model stays where the data is."
  This talks to a *local* Ollama server over plain HTTP, so there are no
  external API keys and no data leaves the machine.
- It works identically on a laptop (Lab 1/2) and in the backup Colab
  instance, because the only thing that changes is the host/port.

The Ollama server is the same one you start in Lab 1 with `ollama serve`
(OpenWebUI also starts it for you). Default endpoint: http://localhost:11434

Set the OLLAMA_HOST environment variable to point somewhere else, e.g.
    export OLLAMA_HOST=http://127.0.0.1:11434      # laptop (default)
    export OLLAMA_HOST=http://localhost:11434       # Colab backup
"""

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _post(path: str, payload: dict, host: str, timeout: int = 600) -> dict:
    url = host.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise SystemExit(
            f"\n[!] Could not reach Ollama at {host}.\n"
            f"    Reason: {e}\n"
            f"    Is the server running? Start it with:  ollama serve\n"
            f"    (OpenWebUI starts it automatically.) Or set OLLAMA_HOST.\n"
        )


def list_models(host: str = DEFAULT_HOST) -> list:
    """Return the tags of models currently pulled locally (Lab 1, step 3)."""
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            tags = json.loads(resp.read().decode("utf-8")).get("models", [])
            return [m["name"] for m in tags]
    except urllib.error.URLError:
        return []


def chat(
    prompt: str,
    model: str = "llama3.2",
    system: str | None = None,
    host: str = DEFAULT_HOST,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> str:
    """
    Send a single-turn prompt to a local model and return the text reply.

    json_mode=True asks Ollama to constrain output to valid JSON, which we use
    for the structured pathology extraction in Lab 3.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    result = _post("/api/chat", payload, host=host)
    return result.get("message", {}).get("content", "").strip()


def ensure_model(model: str, host: str = DEFAULT_HOST) -> bool:
    """Warn (don't crash) if a requested model hasn't been pulled yet."""
    available = list_models(host)
    # Ollama tags include the ':tag' suffix; match on the base name too.
    if any(model == m or model == m.split(":")[0] for m in available):
        return True
    print(
        f"[!] Model '{model}' not found locally. Available: {available or 'none'}\n"
        f"    Pull it first:  ollama pull {model}",
        file=sys.stderr,
    )
    return False
