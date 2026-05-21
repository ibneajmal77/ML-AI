from celery import Celery
from celery.schedules import crontab

from src.config import settings

celery_app = Celery(
    "rag_indexing",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.indexing.tasks"],
)

celery_app.conf.task_routes = {
    "src.indexing.tasks.ingest_document_task": {"queue": "ingest"},
    "src.indexing.tasks.full_reindex_task": {"queue": "reindex"},
    "src.indexing.tasks.incremental_sync_task": {"queue": "ingest"},
}

celery_app.conf.beat_schedule = {
    "full-reindex-weekly": {
        "task": "src.indexing.tasks.full_reindex_task",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2am UTC
    },
    "incremental-sync-daily": {
        "task": "src.indexing.tasks.incremental_sync_task",
        "schedule": crontab(hour=3, minute=0),  # Daily 3am UTC
    },
}
