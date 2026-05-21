from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile

from src.api.deps import get_ingestion_pipeline
from src.domain.models import AccessLevel, DocumentMeta, IngestResponse
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    source_url: str = Form(...),
    owner_dept: str = Form(...),
    access_level: int = Form(default=1),
    tags: str = Form(default=""),
    chunking_strategy: str = Form(default="recursive"),
    use_azure_di: bool = Form(default=False),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    if file.filename is None:
        raise HTTPException(status_code=400, detail="File must have a filename")

    file_bytes = await file.read()
    filename = file.filename
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        access = AccessLevel(access_level)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid access_level: {access_level}")

    meta = DocumentMeta(
        source_url=source_url,
        owner_dept=owner_dept,
        access_level=access,
        tags=tag_list,
    )

    task_id = str(uuid4())
    placeholder_doc_id = uuid4()

    background_tasks.add_task(
        pipeline.run,
        file_bytes,
        filename,
        meta,
        chunking_strategy,
        use_azure_di,
    )

    return IngestResponse(
        document_id=placeholder_doc_id,
        task_id=task_id,
        status="queued",
    )
