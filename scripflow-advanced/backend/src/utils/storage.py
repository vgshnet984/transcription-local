import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from fastapi.responses import FileResponse
from src.config import settings

class LocalFileStorage:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or settings.upload_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = settings.max_file_size_bytes
        self.allowed_exts = set(settings.supported_formats_list)

    def generate_secure_filename(self, filename):
        ext = Path(filename).suffix
        name = secure_filename(Path(filename).stem)
        unique = uuid.uuid4().hex
        return f"{name}_{unique}{ext}"

    def save_file(self, file_obj, filename):
        safe_name = self.generate_secure_filename(filename)
        dest = self.base_dir / safe_name
        with open(dest, "wb") as out:
            shutil.copyfileobj(file_obj, out)
        return str(dest)

    def validate_file(self, path):
        path = Path(path)
        if not path.exists():
            return False, "File not found"
        if path.stat().st_size > self.max_size:
            return False, f"File too large: {path.stat().st_size} bytes"
        if path.suffix.lstrip(".").lower() not in self.allowed_exts:
            return False, f"Unsupported file extension: {path.suffix}"
        return True, None

    def delete_file(self, path):
        path = Path(path)
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_old_files(self, retention_days=30):
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = []
        for f in self.base_dir.iterdir():
            if f.is_file() and datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink()
                deleted.append(str(f))
        return deleted

    def get_file_info(self, path):
        path = Path(path)
        if not path.exists():
            return None
        return {
            "name": path.name,
            "size": path.stat().st_size,
            "created": datetime.fromtimestamp(path.stat().st_ctime),
            "modified": datetime.fromtimestamp(path.stat().st_mtime),
            "path": str(path),
        }

    def serve_file(self, path, download_name=None):
        path = Path(path)
        if not path.exists():
            return None
        return FileResponse(str(path), filename=download_name or path.name) 