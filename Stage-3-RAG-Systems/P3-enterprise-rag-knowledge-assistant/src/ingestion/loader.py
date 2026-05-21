from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    text: str
    metadata: dict
    file_hash: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _load_with_unstructured(file_bytes: bytes, filename: str) -> str:
    from unstructured.partition.auto import partition

    elements = partition(file=io.BytesIO(file_bytes), metadata_filename=filename)

    pages: dict[int, list[str]] = {}
    for el in elements:
        if not el.text or not el.text.strip():
            continue
        page_num = getattr(el.metadata, "page_number", None) or 1
        pages.setdefault(page_num, []).append(el.text.strip())

    parts = []
    for page_num in sorted(pages):
        page_text = "\n".join(pages[page_num])
        parts.append(f"[Page {page_num}]\n{page_text}")

    return "\n\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _load_with_azure_di(file_bytes: bytes) -> str:
    from azure.ai.formrecognizer.aio import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentAnalysisClient(
        endpoint=settings.azure_di_endpoint,
        credential=AzureKeyCredential(settings.azure_di_key),
    )
    async with client:
        poller = await client.begin_analyze_document("prebuilt-layout", file_bytes)
        result = await poller.result()

    parts = []
    for page in result.pages:
        lines = [line.content for line in (page.lines or [])]
        if lines:
            parts.append(f"[Page {page.page_number}]\n" + "\n".join(lines))

    return "\n\n".join(parts)


async def load_document(
    file_bytes: bytes,
    filename: str,
    use_azure_di: bool = False,
) -> RawDocument:
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if use_azure_di:
        text = await _load_with_azure_di(file_bytes)
        source = "azure_di"
    else:
        text = _load_with_unstructured(file_bytes, filename)
        source = "unstructured"

    metadata = {
        "filename": filename,
        "source": source,
        "size_bytes": len(file_bytes),
    }
    return RawDocument(text=text, metadata=metadata, file_hash=file_hash)
