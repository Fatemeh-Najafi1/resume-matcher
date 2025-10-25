import re
#for normalizing text
import nltk
#to tokenize words
from nltk.corpus import stopwords
from config import stopwords_extra
from config import skills_list_path
# you didnt integrate them from  config file in the code yet be careful
#to remove stop words
from nltk.stem import WordNetLemmatizer
#to lemmatize words (turn them into basic form)
# Make sure required NLTK data is downloaded (only once)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class Cleaner:
    #class cleaner is defined with the task to clean and ready
    #the document for embedding
    def __init__(self, stop_words: list,regexe: list ):
        #parameter 
        # 1.stop_word: list of common words in resume
        # not useful for comparison, defined in config file
        self.stop_words = stop_words
        #2.regexe: list of regular experessions not useful for embedding
        #important: extract info from regexe then remove them
        #eg: years of experience from dates, gpa etc
        self.regexe = regexe 
        self.exctracted_info = []
        self.skill_list = []

    def preprocess(self, raw_text: str, doc_type: str):
        #main function of class
        #takes raw text from reader
        self.raw_text = raw_text
        #takes doc type (reume or jobDes)
        self.doc_type = doc_type

        normalized_text = self.normalize(raw_text)
        regexed_text, extracted_info = self.regex_clean(normalized_text)
        #regex_clean has two outputs:1.regexed text 2.extracted info
        lemmatized_text = self.remove_stopwords(regexed_text)
        skill_set = self.extract_skill_set(lemmatized_text)
        final_output = self.create_output(lemmatized_text, extracted_info, skill_set)

        return final_output
        #returns clean text and extraceted values (yoe, gpa etc)

    def normalize(self, raw_text):
        #handles upper lower case, spaces etc
        # 1️⃣ Convert to lowercase
     text = raw_text.lower()

    # 2️⃣ Remove HTML tags or markup (if any)
     text = re.sub(r"<.*?>", " ", text)

    # 3️⃣ Replace line breaks, tabs, and underscores with a space
     text = re.sub(r"[\n\r\t_]+", " ", text)

    # 4️⃣ Remove any remaining non-printable characters
     text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 5️⃣ Collapse multiple spaces into one
     text = re.sub(r"\s+", " ", text)

    # 6️⃣ Strip leading/trailing spaces
     text = text.strip()

     return text
 

    

    def regex_clean(self, normalized_text):
        extracted_info = {}
        text = normalized_text
        #1. remove noise: numbers, dates, personal info
        #2. extract values (year of experience, gpa, etc)
        # Extract GPA
        gpa_match = re.search(r"gpa[:\s]*([0-4]\.\d+)", text, re.IGNORECASE)
        if gpa_match:
         extracted_info["gpa"] = float(gpa_match.group(1))
        text = re.sub(r"gpa[:\s]*([0-4]\.\d+)", " ", text)

    #   Extract years of experience
        years = re.findall(r"(19|20)\d{2}", text)
        if len(years) >= 2:
        # Convert to integers and compute difference
          years = [int(y) for y in years]
          yoe = max(years) - min(years)
          extracted_info["years_of_experience"] = float(yoe)
        else:
         extracted_info["years_of_experience"] = 0.0

    # Remove standalone year references form text
        text = re.sub(r"(19|20)\d{2}", " ", text)

    # 3️⃣ get and remove personal info (emails, phones, URLs)
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " [EMAIL] ", text)
        text = re.sub(r"\b\+?\d[\d\-\s]{7,}\b", " [PHONE] ", text)
        text = re.sub(r"https?://\S+", " [URL] ", text)

    # 4️⃣ Clean leftover punctuation and multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text, extracted_info
        #return:1 regexed text 2.exctracted info
        

    def remove_stopwords(self, regexed_text):
        #turn text into words(tokens
        tokens = re.findall(r'\b\w+\b', regexed_text)
        #removes common words that are unnecesarry for comparison
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        #lemmatize
        lemmitizer = WordNetLemmatizer()
        lemmitized = [lemmitizer.lemmatize(word) for word in filtered_tokens]
        #rejoin words
        lemmitized_text = " ".join(lemmitized)
        return lemmitized_text
        

    def extract_skill_set(self, lemmatized_text):
        words = lemmatized_text.split()
        #create a list of skills
        extracted_skills = [word for word in words if word in self.skill_list]
        #removig duplicates
        extracted_skills = list(set(extracted_skills))

        #return skill set
        return extracted_skills

    def create_output(self, lemma_text , extract_info ,skill_set):
        #returns output:
        #1.clean text 2.skill list 3.extracted info
       
         return {
            "clean_text": lemma_text,
            "skills": skill_set,
            "extracted_info": extract_info,
         }
        
