# src/config.py

import os

# 🔹 Stopword extensions (custom for your domain)
stopwords_extra = [
    "experience", "skills", "responsible", "work", "developed", "project", "resume"
]

# 🔹 Skills list file path (you can replace with an actual .txt file later)
skills_list_path = os.path.join("data", "skills_list.txt")

# 🔹 Threshold for similarity
THRESHOLD = 0.7