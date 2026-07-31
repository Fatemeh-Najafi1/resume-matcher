# src/Reader.py
import os
import tempfile

import docx2txt
import pdfplumber

# libraries:
# 1. os / tempfile → file paths, extensions, and a scratch dir for uploads
# 2. docx2txt      → lightweight text extraction from .docx (Word) files
# 3. pdfplumber    → reads PDFs and extracts text page by page


class UnreadableDocumentError(ValueError):
    """Raised when a file opens fine but yields no usable text.

    The common case is a scanned PDF: the pages are images, so there is no
    text layer to extract. Previously this returned an empty string, which
    flowed silently through the pipeline and surfaced as a near-zero match
    score with no explanation. Failing loudly here is the whole point.
    """


class Reader:
    def __init__(self, tmp_dir=None):
        """
        tmp_dir: scratch directory for web uploads, which arrive as file-like
        objects and have to be written to disk before pdfplumber/docx2txt can
        open them. Defaults to the OS temp dir (the old "/tmp" default created
        a stray C:\\tmp on Windows).
        """
        self.tmp_dir = tmp_dir or os.path.join(tempfile.gettempdir(), "resume_matcher")
        os.makedirs(self.tmp_dir, exist_ok=True)

    # 1. txt format
    def _read_txt(self, path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # 2. docx format
    def _read_docx(self, path):
        return docx2txt.process(path) or ""

    # 3. pdf format
    def _read_pdf(self, path):
        pages = []
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                pages.append(page.extract_text() or "")

        text = "\n".join(pages)
        if not text.strip():
            raise UnreadableDocumentError(
                f"This PDF has no extractable text ({page_count} page(s) checked). "
                "It's most likely a scan or a photo of a document — the pages are "
                "images, not text. Export a text-based PDF, or paste the text directly."
            )
        return text

    # web uploads
    def _is_filelike(self, obj):
        """Check if input is a file-like object (i.e. a web upload)."""
        return hasattr(obj, "read")

    def read_file(self, input_obj):
        """
        Reads a file (path or file-like) and returns extracted text.
        Supports TXT, DOCX, PDF.
        """
        # file-like object (from a web upload) → spool it to disk first
        if self._is_filelike(input_obj):
            filename = os.path.basename(getattr(input_obj, "name", "uploaded_file"))
            data = input_obj.read()
            if not data:
                raise UnreadableDocumentError(f"'{filename}' is empty.")

            path = os.path.join(self.tmp_dir, filename)
            with open(path, "wb") as tmp:
                tmp.write(data)
            input_obj = path  # continue as a normal path

        path = str(input_obj)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".txt":
            text = self._read_txt(path)
        elif ext == ".docx":
            text = self._read_docx(path)
        elif ext == ".pdf":
            text = self._read_pdf(path)
        else:
            raise ValueError(
                f"Unsupported file format: '{ext or path}'. Use TXT, DOCX or PDF."
            )

        if not text.strip():
            raise UnreadableDocumentError(
                f"No text could be read from '{os.path.basename(path)}'."
            )
        return text

    # read resume
    def load_resume(self, path_or_file):
        return self.read_file(path_or_file)

    # read job
    def load_job_description(self, path_or_file):
        return self.read_file(path_or_file)
