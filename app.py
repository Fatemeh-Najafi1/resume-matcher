# app.py
import argparse
import os
import sys

from src.Reader import Reader, UnreadableDocumentError
from src.Cleaner import Cleaner
from src.Embedder import Embedder
from src.Comparator import Comparator
from src.Generator import Generator


def read_or_prompt(reader, path, label):
    """Load a document from disk, or fall back to pasted text."""
    if path and os.path.exists(path):
        try:
            return reader.read_file(path)
        except (UnreadableDocumentError, ValueError) as e:
            print(f"\n⚠️ Could not read {label}: {e}")
    else:
        print(f"\n⚠️ {label.capitalize()} file not found or not provided.")
    return input(f"Please paste the {label} text below:\n> ")


def main():
    parser = argparse.ArgumentParser(description="Resume Matcher")
    parser.add_argument("--resume", help="Path to resume file")
    parser.add_argument("--job", help="Path to job description file")
    args = parser.parse_args()

    reader = Reader()

    # 1. Load both documents
    resume_text = read_or_prompt(reader, args.resume, "resume")
    job_text = read_or_prompt(reader, args.job, "job description")

    if not resume_text.strip() or not job_text.strip():
        print("Both a resume and a job description are required.")
        sys.exit(1)

    # 2. Preprocess — produces clean_text for embedding and keywords for the gap report
    c = Cleaner()
    resume_clean = c.preprocess(resume_text, doc_type="resume")
    job_clean = c.preprocess(job_text, doc_type="job")

    # 3. Embeddings — text becomes a 384-dim vector
    # Picks the multilingual encoder if either document contains Arabic.
    e = Embedder.for_documents(resume_text, job_text)
    resume_vect = e.get_embeddings(resume_clean["clean_text"])
    job_vect = e.get_embeddings(job_clean["clean_text"])

    # 4. Compare — cosine similarity between the two vectors
    comp = Comparator(resume_text, job_text, model="basic")
    score = comp.calculate_similarity(resume_vect, job_vect)
    insight = comp.provide_insight(resume_vect, job_vect, score)

    # 5. Keyword gap report (literal comparison, no model involved)
    report = Generator().generate_text(resume_clean, job_clean, score)

    # 6. Output
    print(f"\nSimilarity score: {round(score * 100, 1)}%")
    print(f"Insight: {insight['interpretation']}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
