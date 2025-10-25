# app.py
import argparse
import os
from src.Reader import Reader
from src.Cleaner import Cleaner
from src.Embedder import Embedder
from src.Comparator import Comperator
from src.Generator import Generator
# from src.utils import print_summary

def main():
    reader = Reader()
    parser = argparse.ArgumentParser(description="Resume App")
    parser.add_argument("--resume", help="Path to resume file")  
    parser.add_argument("--job", help="Path to job description file")  
    args = parser.parse_args()

    # 1. Load files(resume+job description)
    if args.resume and os.path.exists(args.resume):
        resume_text = reader.load_resume(args.resume)
    else:
        print("\n⚠️ Resume file not found or not provided.")
        resume_text = input("Please paste your resume text below:\n> ")

    if args.job and os.path.exists(args.job):
        job_text = reader.load_job_description(args.job)
    else:
        print("\n⚠️ Job description file not found or not provided.")
        job_text = input("Please paste the job description text below:\n> ")

    # 2. Preprocess (cleans text for embedding)
    c = Cleaner(stop_words=[], regexe=[])  # (create cleaner instance)
    resume_clean = c.preprocess(resume_text, doc_type="resume")  # calling insatnce
    job_clean = c.preprocess(job_text, doc_type="job")  

    # 3. Embeddings (turns text into vector for matching)
    e = Embedder(model="basic")  # create embedder instance)
    #resume_vect = e.get_embeddings(resume_clean, model="basic")  # insyance  call)
    #job_vect = e.get_embeddings(job_clean, model="basic")  # instance call)
    # 3️⃣ Embeddings (turns text into vector for matching)
    resume_vect = e.get_embeddings(resume_clean["clean_text"], model="basic")
    job_vect = e.get_embeddings(job_clean["clean_text"], model="basic")


    # 4. Compare similarity
    comp = Comperator(resume_text, job_text, model="basic")  #create comparator instance)
    score = comp.calculate_similarity(resume_vect, job_vect)  # instance method)
    summar = comp.provide_insight(resume_vect, job_vect, score)  #added score argument)

    # 5. Generate rewritten resume
    gen = Generator()  # create generator instance)
    rewritten_resume = gen.generate_text(resume_text, job_text, score)  #instance method)

    # 6. Output
    
    print("\nSimilarity Score:", score)
    print(" Insight:", summar)
    print("\n Rewritten Resume:\n", rewritten_resume)


if __name__ == "__main__":
    main()

