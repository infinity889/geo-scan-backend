import shutil
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from app.core.config import settings
from app.domain.models import DocumentSummary, PipelineStep, StepStatus
from app.services.seed import STEP_LOGS, initial_steps


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 / 1024:.1f} MB"


def detect_format(file_name: str) -> str:
    suffix = Path(safe_filename(file_name)).suffix.replace(".", "").upper()
    return suffix or "FILE"


def safe_filename(file_name: str) -> str:
    return Path(PureWindowsPath(file_name).name).name


class InMemoryGeoStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._documents: OrderedDict[str, DocumentSummary] = OrderedDict()
        self.upload_dir = settings.upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._seed()

    def _seed(self) -> None:
        for item in [
            ("doc-1", "Priobskoye_Report_2015.pdf", 8_808_038, "PDF"),
            ("doc-2", "Well_247_Core_Analysis.docx", 2_202_010, "DOCX"),
        ]:
            doc_id, name, size_bytes, file_format = item
            steps = [
                step.model_copy(
                    update={"status": "completed", "log": STEP_LOGS.get(step.id, [])}
                )
                for step in initial_steps()
            ]
            self._documents[doc_id] = DocumentSummary(
                id=doc_id,
                name=name,
                size=human_size(size_bytes),
                size_bytes=size_bytes,
                format=file_format,
                status="completed",
                created_at=datetime.now(timezone.utc),
                steps=steps,
            )

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            return list(self._documents.values())

    def get_document(self, document_id: str) -> DocumentSummary | None:
        with self._lock:
            return self._documents.get(document_id)

    def save_upload(self, file_name: str, stream: BinaryIO) -> DocumentSummary:
        document_id = f"doc-{uuid.uuid4().hex[:8]}"
        display_name = safe_filename(file_name)
        target = self.upload_dir / f"{document_id}-{display_name}"
        with target.open("wb") as handle:
            shutil.copyfileobj(stream, handle)

        size_bytes = target.stat().st_size
        doc = DocumentSummary(
            id=document_id,
            name=display_name,
            size=human_size(size_bytes),
            size_bytes=size_bytes,
            format=detect_format(file_name),
            status="pending",
            created_at=datetime.now(timezone.utc),
            steps=initial_steps(),
        )

        with self._lock:
            self._documents[document_id] = doc

        thread = threading.Thread(
            target=self._simulate_pipeline,
            args=(document_id,),
            daemon=True,
        )
        thread.start()
        return doc

    def _simulate_pipeline(self, document_id: str) -> None:
        import asyncio
        from app.services.processor import process_document_async

        with self._lock:
            doc = self._documents.get(document_id)
            if not doc:
                return
            document_name = doc.name

        file_path = self.upload_dir / f"{document_id}-{document_name}"

        self._update_step(document_id, 1, status="processing", log=["Started RAG pipeline"])
        self._update_step(document_id, 2, status="processing", log=["Awaiting extraction"])

        def log_cb(msg: str):
            self._append_step_log(document_id, 2, msg)

        asyncio.run(process_document_async(document_id, document_name, file_path, log_cb))

        for step_id in [1, 2, 3, 4]:
            self._update_step(document_id, step_id, status="completed")
        self._update_document_status(document_id, "completed")

    def _update_document_status(self, document_id: str, status: StepStatus) -> None:
        with self._lock:
            doc = self._documents.get(document_id)
            if doc is None:
                return
            self._documents[document_id] = doc.model_copy(update={"status": status})

    def _update_step(
        self,
        document_id: str,
        step_id: int,
        status: StepStatus,
        log: list[str] | None = None,
    ) -> None:
        with self._lock:
            doc = self._documents.get(document_id)
            if doc is None:
                return
            steps = [
                self._patch_step(step, status, log)
                if step.id == step_id
                else step
                for step in doc.steps
            ]
            aggregate_status = "processing" if status == "processing" else doc.status
            self._documents[document_id] = doc.model_copy(
                update={"steps": steps, "status": aggregate_status}
            )

    def _append_step_log(self, document_id: str, step_id: int, line: str) -> None:
        with self._lock:
            doc = self._documents.get(document_id)
            if doc is None:
                return
            steps = [
                step.model_copy(update={"log": [*step.log, line]})
                if step.id == step_id
                else step
                for step in doc.steps
            ]
            self._documents[document_id] = doc.model_copy(update={"steps": steps})

    @staticmethod
    def _patch_step(
        step: PipelineStep,
        status: StepStatus,
        log: list[str] | None,
    ) -> PipelineStep:
        update: dict[str, object] = {"status": status}
        if log is not None:
            update["log"] = log
        return step.model_copy(update=update)


store = InMemoryGeoStore()
