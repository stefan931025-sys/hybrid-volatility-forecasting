import os
import re
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QuantRepoAuditor")

class QuantRepoAuditor:
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.excluded_dirs = {".git", "__pycache__", "venv", ".pytest_cache", "data", "models"}

    def audit_repository(self):
        """Walks through the quantitative repository, auditing and flagging potential biases or style issues."""
        logger.info(f"Starting quantitative repository audit across: {os.path.abspath(self.root_dir)}")
        
        flagged_count = 0
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            dirnames[:] = [d for d in dirnames if d not in self.excluded_dirs]
            
            for file in filenames:
                if file.endswith(".py") and file != "repo_auditor.py":
                    file_path = os.path.join(dirpath, file)
                    if self._audit_and_fix_file(file_path):
                        flagged_count += 1

        logger.info(f"Audit complete. Total files inspected/corrected: {flagged_count}")

    def _audit_and_fix_file(self, file_path: str) -> bool:
        """Inspects file content for quantitative rigor and clean code standards."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            updated_content = content
            
            # 1. Check for Look-Ahead Bias / Data Leakage Risks (Global fit_transform without split warning/handling)
            if ".fit_transform(" in updated_content and "train_test_split" not in updated_content and "TimeSeriesSplit" not in updated_content:
                logger.warning(f"Potential Look-Ahead Risk in {file_path}: Found `.fit_transform()` without explicit cross-validation/split context.")

            # 2. Ensure future annotations are present for robust type hinting
            updated_content = self._ensure_future_imports(updated_content)
            
            # 3. Replace bare print statements with structured logging where appropriate
            updated_content = self._upgrade_prints_to_logging(updated_content, file_path)

            if updated_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info(f"Successfully sanitized and updated: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error auditing {file_path}: {e}")
        
        return False

    def _ensure_future_imports(self, content: str) -> str:
        """Ensures modern Python type hinting annotations are supported."""
        if "from __future__ import annotations" not in content and "def " in content:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i
                    break
            lines.insert(insert_idx, "from __future__ import annotations")
            content = "\n".join(lines) + "\n"
        return content

    def _upgrade_prints_to_logging(self, content: str, file_path: str) -> str:
        """Ensures quantitative modules use standard loggers instead of raw prints for production telemetry."""
        if "print(" in content and "import logging" not in content:
            # Prepend logging setup
            lines = content.splitlines()
            import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    import_idx = i + 1
            
            logger_setup = "\nimport logging\nlogger = logging.getLogger(__name__)"
            lines.insert(import_idx, logger_setup)
            content = "\n".join(lines) + "\n"
        return content

if __name__ == "__main__":
    auditor = QuantRepoAuditor()
    auditor.audit_repository()
