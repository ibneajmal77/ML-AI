import json
import os
import sys
from pathlib import Path

from openai import AzureOpenAI, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def create_client() -> tuple[OpenAI | AzureOpenAI, str]:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if api_key:
        return OpenAI(api_key=api_key), model

    azure_key = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if azure_key and azure_endpoint and azure_deployment:
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
        return client, azure_deployment

    raise RuntimeError(
        "Set OPENAI_API_KEY or AZURE_OPENAI_KEY/AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT "
        "in the environment or .env file."
    )


client, MODEL = create_client()

EXTRACT_PROMPT = """Extract the following ticket into JSON with exactly these keys:
issue_type, account_id, urgency, requested_action.
Use null for missing values. Return JSON only.

Ticket:
The customer says their board meeting is in two hours, they cannot log in,
and their account ID is ACC-20491. They want immediate password reset help.
"""


def extract_at_max_tokens(max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": EXTRACT_PROMPT}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


print(f"{'max_tokens':>12}  {'parse result':<25}  raw (first 120 chars)")
print("-" * 80)

for limit in [20, 50, 100, 150, 300]:
    raw = extract_at_max_tokens(limit)
    try:
        json.loads(raw)
        parse_result = "valid JSON"
    except json.JSONDecodeError:
        parse_result = "invalid / truncated"
    snippet = raw.replace("\n", " ")[:120]
    print(f"{limit:>12}  {parse_result:<25}  {snippet}")
