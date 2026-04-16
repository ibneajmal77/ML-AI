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

SYSTEM = (
    "You are a support ticket classifier. Return exactly one lowercase label: "
    "billing, technical, account, or general. Return only the label."
)
TICKET = "I got charged $49.99 last week but I cancelled my subscription two months ago."


def classify_at_temp(temperature: float, runs: int = 5) -> list[str]:
    results: list[str] = []
    for _ in range(runs):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TICKET},
            ],
            temperature=temperature,
            max_tokens=10,
            stop=["\n"],
        )
        content = response.choices[0].message.content or ""
        results.append(content.strip().lower())
    return results


for temp in [0.0, 0.7, 1.5]:
    labels = classify_at_temp(temp)
    print(f"temperature={temp}: {labels} -> unique labels: {sorted(set(labels))}")
