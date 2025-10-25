# src/reader.py
import os
import docx2txt
import pdfplumber
#libraries: 
# 1.os → to manipulate file paths and extensions 

#2.docx2txt → lightweight library to extract text from .docx (Word) files.

#3. pdfplumber → reads PDFs and extracts text page by page 
 

class Reader:
  def __init__(self, tmp_dir="/tmp"):
      """
      Initialize Reader instance.
      tmp_dir: Temporary directory for handling file uploads or conversions.
      """
      self.tmp_dir = tmp_dir
      if not os.path.exists(tmp_dir):
          os.makedirs(tmp_dir)

  # 1. txt format
  def _read_txt(self, path):
      with open(path, "r", encoding="utf-8") as f:
          return f.read()

  #2. docx format
  def _read_docx(self, path):
      return docx2txt.process(path)

  #pdf format
  def _read_pdf(self, path):
      text = ""
      with pdfplumber.open(path) as pdf:
          for page in pdf.pages:
              text += page.extract_text() or ""
      return text

  #web uplouds 
  def _is_filelike(self, obj):
      #Check if input is a file-like object (if its web uploads)
      return hasattr(obj, "read")

  def read_file(self, input_obj):
      """
      Reads a file (path or file-like) and returns extracted text.
      Supports TXT, DOCX, PDF.
      """
      # if file-like object (from web upload)
      if self._is_filelike(input_obj):
          # safely get filename and extension
          filename = getattr(input_obj, "name", "uploaded_file")
          ext = os.path.splitext(filename)[1].lower()
          
          # safely read file bytes
          data = input_obj.read() if hasattr(input_obj, "read") else None
          path = os.path.join(self.tmp_dir, filename)

          # write temporary file
          if data:
              with open(path, "wb") as tmp:
                  tmp.write(data)

          input_obj = path  # rewrite to file path

      # handle local file path
      filename = getattr(input_obj, "name", str(input_obj))
      ext = os.path.splitext(filename)[1].lower()

      if ext == ".txt":
          return self._read_txt(input_obj)
      elif ext == ".docx":
          return self._read_docx(input_obj)
      elif ext == ".pdf":
          return self._read_pdf(input_obj)
      else:
          raise ValueError(f"Unsupported file format: {ext}")


  # read resume
  def load_resume(self, path_or_file):
      return self.read_file(path_or_file)

  # read job
  def load_job_description(self, path_or_file):
      return self.read_file(path_or_file)
