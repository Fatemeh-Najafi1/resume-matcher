import re
# for normalizing text
import nltk
# to tokenize words
from nltk.corpus import stopwords as nltk_stopwords
# to remove stop words
from nltk.stem import WordNetLemmatizer
# to lemmatize words (turn them into basic form)

from .config import STOPWORDS_EXTRA
from .utils import skill_patterns


def _ensure_nltk(resource, package):
    """Download an NLTK corpus once, the first time it's needed."""
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


_ensure_nltk("corpora/stopwords", "stopwords")
_ensure_nltk("corpora/wordnet", "wordnet")


class Cleaner:
    """Prepares raw document text for the rest of the pipeline.

    Produces two views of the same document, because the two consumers want
    different things:

      * clean_text — normalized and lemmatized, stopwords KEPT. This goes to
        the Embedder. all-MiniLM-L6-v2 was trained on natural sentences, so
        stripping "and"/"with"/"of" degrades the vector rather than sharpening it.
      * keywords — the same text with stopwords removed. This goes to the
        Generator, where the comparison is literal set difference and filler
        words would otherwise dominate the gap report.
    """

    def __init__(self, stop_words=None, regexe=None, skills=None):
        # 1. stop_words: words too common to carry signal. Defaults to NLTK's
        #    English list plus the resume boilerplate in config.yaml. Pass an
        #    explicit list (or []) to override, which is handy in tests.
        if stop_words is None:
            stop_words = set(nltk_stopwords.words("english")) | set(STOPWORDS_EXTRA)
        self.stop_words = set(stop_words)

        # 2. regexe: reserved for caller-supplied extraction patterns.
        self.regexe = regexe or []

        # 3. skills: vocabulary to match against, from data/skills_list.txt.
        # Shared with Matcher via utils, so both judge "does this text mention
        # PyTorch" by the same rule. A caller can still pass its own list.
        if skills is None:
            self.skill_patterns = list(skill_patterns())
        else:
            self.skill_patterns = [
                (s.lower(), re.compile(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])"))
                for s in skills
            ]

        self.lemmatizer = WordNetLemmatizer()

    # Legal/HR sections that appear in most postings and describe the employer's
    # obligations, not the role. Left in, they flood the gap report with words
    # like "accommodation", "applicant" and "age", and dilute the embedding.
    BOILERPLATE_MARKERS = (
        "equal opportunity employer",
        "equal employment opportunity",
        "we are an equal",
        "affirmative action",
        "reasonable accommodation",
        "without regard to race",
        "we do not accept unsolicited",
        "privacy policy",
        "e-verify",
    )

    def strip_boilerplate(self, text):
        """Cut the closing legal block off a job posting.

        Only trims from the halfway point onward — a posting that happens to
        mention equal opportunity in its opening paragraph keeps its content.
        """
        lowered = text.lower()
        cut = len(text)
        for marker in self.BOILERPLATE_MARKERS:
            idx = lowered.find(marker)
            if idx > len(text) * 0.5:
                cut = min(cut, idx)
        return text[:cut].strip()

    def preprocess(self, raw_text: str, doc_type: str):
        """Main entry point: raw text in, structured views out."""
        self.raw_text = raw_text
        self.doc_type = doc_type

        if doc_type == "job":
            raw_text = self.strip_boilerplate(raw_text)

        normalized_text = self.normalize(raw_text)
        regexed_text, extracted_info = self.regex_clean(normalized_text)
        # skills are matched before stopword removal so phrases stay intact
        skill_set = self.extract_skill_set(regexed_text)
        lemmatized_text = self.lemmatize(regexed_text)
        keyword_counts = self.extract_keywords(lemmatized_text)

        return self.create_output(lemmatized_text, keyword_counts, extracted_info, skill_set)

    def normalize(self, raw_text):
        """Handle case, whitespace, markup and non-printable characters."""
        # 1. Convert to lowercase
        text = raw_text.lower()
        # 2. Remove HTML tags or markup (if any)
        text = re.sub(r"<.*?>", " ", text)
        # 3. Replace line breaks, tabs and underscores with a space
        text = re.sub(r"[\n\r\t_]+", " ", text)
        # 4. Remove any remaining non-printable characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # 5. Collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text)
        # 6. Strip leading/trailing spaces
        return text.strip()

    def regex_clean(self, normalized_text):
        """Pull structured facts out, then remove the noise they came from."""
        extracted_info = {}
        text = normalized_text

        # GPA
        gpa_match = re.search(r"gpa[:\s]*([0-4]\.\d+)", text)
        if gpa_match:
            extracted_info["gpa"] = float(gpa_match.group(1))
        text = re.sub(r"gpa[:\s]*([0-4]\.\d+)", " ", text)

        # Years of experience, inferred from the span of years mentioned
        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)]
        if len(years) >= 2:
            extracted_info["years_of_experience"] = float(max(years) - min(years))
        else:
            extracted_info["years_of_experience"] = 0.0

        # Remove standalone year references from the text
        text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)

        # Personal info (emails, phones, URLs) — replaced with placeholders
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " [email] ", text)
        text = re.sub(r"\b\+?\d[\d\-\s]{7,}\b", " [phone] ", text)
        text = re.sub(r"https?://\S+", " [url] ", text)

        # Any extra caller-supplied patterns
        for pattern in self.regexe:
            text = re.sub(pattern, " ", text)

        # Collapse the gaps the substitutions left behind
        text = re.sub(r"\s+", " ", text).strip()

        return text, extracted_info

    def _lemma(self, word):
        """Lemmatize as a verb first, then as a noun.

        WordNetLemmatizer defaults to pos="n", which leaves every verb form
        untouched — "managed" stays "managed". Since resumes are written almost
        entirely in past-tense verbs, the noun-only default did essentially
        nothing. Running the verb pass first fixes "managed" -> "manage", and
        the noun pass still handles plurals like "skills" -> "skill".
        """
        return self.lemmatizer.lemmatize(self.lemmatizer.lemmatize(word, pos="v"), pos="n")

    def lemmatize(self, regexed_text):
        """Reduce words to their base form so 'managed' and 'manage' agree.

        Stopwords are deliberately kept here — see the class docstring.
        """
        tokens = re.findall(r"\b\w+\b", regexed_text)
        return " ".join(self._lemma(word) for word in tokens)

    def extract_keywords(self, lemmatized_text):
        """Content words and how often each occurs.

        The counts matter: a term the posting repeats is far more likely to be
        a real requirement than one mentioned once in passing. The gap report
        ranks by that instead of alphabetically, which previously buried the
        useful terms under "accelerate, accommodation, achieve, across".
        """
        tokens = re.findall(r"\b\w+\b", lemmatized_text)
        counts = {}
        for word in tokens:
            if word in self.stop_words or len(word) <= 2 or word.isdigit():
                continue
            counts[word] = counts.get(word, 0) + 1
        return counts

    def extract_skill_set(self, text):
        """Match the document against the known skill vocabulary.

        Runs on the pre-lemmatization text on purpose: lemmatizing would turn
        "machine learning" into "machine learn" and the phrase would stop
        matching the vocabulary.
        """
        return sorted(skill for skill, pattern in self.skill_patterns if pattern.search(text))

    def create_output(self, lemma_text, keyword_counts, extract_info, skill_set):
        return {
            "clean_text": lemma_text,             # -> Embedder (stopwords kept)
            "keywords": set(keyword_counts),      # -> Generator (stopwords removed)
            "keyword_counts": keyword_counts,     # -> Generator, for ranking
            "skills": skill_set,
            "extracted_info": extract_info,
        }
