# Resume Matcher

A Streamlit app that compares a resume against a job description using NLP embeddings, computes a semantic similarity score, and identifies skill gaps — without fabricating skills you don't have.

🔗 **[Live Demo](https://resume-m-nkdtv7rpdzsjvtktlqx5zt.streamlit.app/)**

---

## The Problem This Solves

Most resume tools either score your resume blindly or suggest adding skills you don't actually have. This tool takes a different approach: it tells you how well you genuinely match a role, surfaces the specific keywords and skills you're missing, and explicitly flags which ones you should develop before claiming them — not add dishonestly.

---

## What It Does

- Accepts a resume and job description in **TXT, DOCX, or PDF** format
- Cleans and preprocesses both documents (normalization, lemmatization, noise removal)
- Extracts structured metadata: GPA, years of experience, contact info
- Embeds both documents using **sentence-transformers** (`all-MiniLM-L6-v2`)
- Computes **cosine similarity** to produce a match score (0–100%)
- Provides a plain-English assessment of your fit for the role
- Identifies keywords present in the job description but missing from your resume
- Suggests which terms to add (if you have the experience) and which to develop first

---

## Architecture

```
Input (TXT / DOCX / PDF)
        ↓
    Reader        → loads resume and job description
        ↓
    Cleaner       → normalizes, lemmatizes, extracts GPA / YoE
        ↓
    Embedder      → sentence-transformers (all-MiniLM-L6-v2)
        ↓
    Comparator    → cosine similarity + score interpretation
        ↓
    Generator     → keyword gap analysis + match report
        ↓
  Streamlit UI    → score, insight, suggested resume updates
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| UI | Streamlit |
| NLP / Embeddings | sentence-transformers, NLTK |
| File parsing | pdfplumber, docx2txt |
| ML | scikit-learn, NumPy |
| Config | YAML |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Fatemeh-Najafi1/resume-matcher.git
cd resume-matcher

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app_streamlit.py
```

Or use the CLI version:
```bash
python app.py --resume your_resume.txt --job job_description.txt
```

---

## Project Structure

```
resume-matcher/
├── app.py                  # CLI entry point
├── app_streamlit.py        # Streamlit UI
├── config.py               # Configuration constants
├── config.yaml             # Model and threshold settings
├── requirements.txt
├── src/
│   ├── Reader.py           # File loader (TXT, DOCX, PDF)
│   ├── Cleaner.py          # Text preprocessing pipeline
│   ├── Embedder.py         # Embedding models
│   ├── Comparator.py       # Similarity scoring + insight
│   └── Generator.py        # Keyword gap analysis + report
└── data/                   # Skills lists and config data
```

---

## Roadmap

- [ ] Integrate OpenAI / Ollama for LLM-powered resume rewriting
- [ ] Scrape job descriptions directly from LinkedIn or company websites
- [ ] Skill gap categorization (have it vs. need to learn it)
- [ ] Multi-resume comparison against a single job description
- [ ] Export tailored resume as PDF or DOCX

---

## Author

**Fatemeh Najafi** — [GitHub](https://github.com/Fatemeh-Najafi1) · [LinkedIn](https://www.linkedin.com/in/fatemeh-najafi-797555382)
