"""Upload extraction helpers for API routes."""

from typing import Optional

from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile


def extract_uploaded_file(audio_file: UploadFile | None, file: UploadFile | None, form_data: dict) -> UploadFile | None:
    """Extract uploaded file from route parameters and form data."""
    uploaded_file = _extract_upload_from_primary_params(audio_file, file)
    if uploaded_file:
        return uploaded_file
    uploaded_file = _extract_upload_from_named_form_fields(form_data)
    if uploaded_file:
        return uploaded_file
    return _find_upload_file_in_dict(form_data)


def _extract_upload_from_primary_params(audio_file, file) -> Optional[UploadFile]:
    uploaded_file = audio_file or file
    if _is_valid_upload_file(uploaded_file):
        return uploaded_file
    return None


def _extract_upload_from_named_form_fields(form_data: dict) -> Optional[UploadFile]:
    uploaded_file = form_data.get("audio_file") or form_data.get("file") or form_data.get("video_file")
    if _is_valid_upload_file(uploaded_file):
        return uploaded_file
    return None


def _is_valid_upload_file(val) -> bool:
    return val is not None and isinstance(val, (UploadFile, StarletteUploadFile))


def _find_upload_file_in_dict(data: dict) -> Optional[UploadFile]:
    for value in data.values():
        if isinstance(value, (UploadFile, StarletteUploadFile)):
            return value
    return None
