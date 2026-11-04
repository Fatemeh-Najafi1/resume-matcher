# tests/test_verdict.py
"""The distinction this project exists to make, pinned down.

Every test here is a claim the README makes. If one of them fails, the README
is lying, which is a worse bug than a wrong score.
"""

import pytest

from src import Verdict
from src.Matcher import hard_gate, match, split_bullets
from src.utils import degree_level, degree_required, find_skills, normalise


# ── Silence is not disproof ──────────────────────────────────────────────────

class TestSilenceIsNotDisproof:
    """A resume that does not mention X says nothing about X.

    This is the whole thesis. Collapsing "not mentioned" into "does not have"
    is what filters out qualified people for an omission.
    """

    def test_unmentioned_skill_is_unsupported_not_contradicted(self):
        status, _, _ = hard_gate("Experience with Kafka",
                                 "Python, Pandas, scikit-learn, Streamlit")
        assert status == Verdict.UNSUPPORTED
        assert status not in Verdict.BLOCKING

    def test_no_degree_stated_is_unsupported(self):
        status, reason, _ = hard_gate("Master's degree in Computer Science",
                                      "Built ML pipelines. Shipped a Streamlit app.")
        assert status == Verdict.UNSUPPORTED
        assert "does not state a degree" in reason

    def test_lower_degree_stated_is_contradicted(self):
        status, _, _ = hard_gate("Master's degree in Computer Science",
                                 "B.Sc. Computer Science, 2024")
        assert status == Verdict.CONTRADICTED
        assert status in Verdict.BLOCKING

    def test_similarity_path_never_contradicts(self):
        """Cosine can fail to find evidence; it cannot establish a conflict."""
        status, _, _ = hard_gate("Ability to work on ambiguous problems",
                                 "Python, Pandas, scikit-learn")
        # No named skill and no degree, so the gate defers entirely.
        assert status is None


# ── Degree levels are ordinal, and requirements state a floor ────────────────

class TestDegreeLevels:

    def test_higher_degree_satisfies_lower_demand(self):
        status, _, _ = hard_gate("Bachelor's degree required", "M.Sc. Data Science")
        assert status == Verdict.COVERED

    @pytest.mark.parametrize("requirement", [
        "PhD or Master's degree preferred",
        "Master's or PhD",
        "Master's degree required, PhD preferred",
    ])
    def test_disjunctive_demand_takes_the_lowest_level(self, requirement):
        """'Master's or PhD' is satisfied by a master's.

        Reading the highest level named instead told every qualified
        master's-holder they were contradicted by a posting that would have
        hired them.
        """
        assert degree_required(requirement) == 2
        status, _, _ = hard_gate(requirement, "M.Sc. Software Engineering")
        assert status == Verdict.COVERED

    def test_resume_degree_takes_the_highest_level(self):
        """A resume naming both means the higher one is held."""
        assert degree_level("B.Sc. 2018, then M.Sc. Data Science 2021") == 2


# ── Durations ────────────────────────────────────────────────────────────────

class TestDuration:

    def test_stated_shortfall_contradicts(self):
        status, reason, _ = hard_gate(
            "5+ years of experience with Python",
            "Python, Django. 3 years of experience in backend development.")
        assert status == Verdict.CONTRADICTED
        assert "3" in reason

    def test_unstated_duration_is_partial_not_contradicted(self):
        """Naming a tool never proves a duration — but it never disproves one."""
        status, _, _ = hard_gate("5+ years of experience with Python",
                                 "Python, Django, Flask. Built production services.")
        assert status == Verdict.PARTIAL

    def test_sufficient_date_span_covers(self):
        status, _, _ = hard_gate("2 years of experience with Python",
                                 "Python developer, 2019 - 2026.")
        assert status == Verdict.COVERED

    def test_small_date_span_shortfall_does_not_contradict(self):
        """Date spans are noisy, so a near miss stays partial.

        An education start year inflates a span and a career gap deflates it,
        so only a shortfall wider than the margin is called a conflict.
        """
        status, _, _ = hard_gate("4 years of Python", "Python developer, 2023 - 2026.")
        assert status == Verdict.PARTIAL


# ── Arabic ───────────────────────────────────────────────────────────────────

class TestArabicNormalisation:
    """Arabic has several legal spellings of the same word.

    Matching literally without folding them is a silent failure: the match
    simply never fires and nothing reports that it didn't.
    """

    @pytest.mark.parametrize("variant", ["الذكاء الإصطناعي", "الذكاء الاصطناعي"])
    def test_alef_hamza_variants_fold_together(self, variant):
        assert normalise(variant) == normalise("الذكاء الاصطناعي")

    def test_tatweel_is_stripped(self):
        assert normalise("تحليـــل") == normalise("تحليل")

    def test_teh_marbuta_folds_to_heh(self):
        assert normalise("البرمجة") == normalise("البرمجه")

    def test_arabic_indic_digits_become_ascii(self):
        assert "2026" in normalise("٢٠٢٦")
        assert "5" in normalise("٥ سنوات")

    def test_arabic_years_parse(self):
        assert Verdict.years_required("خبرة لا تقل عن ٥ سنوات") == 5.0

    def test_word_boundaries_hold_in_arabic(self):
        """`(?<![a-z0-9])` does nothing in Arabic script.

        An Arabic letter is not in [a-z0-9], so the old lookaround passed and
        a vocabulary term matched happily inside a longer Arabic word.
        """
        from src.utils import _bounded
        pattern = _bounded(normalise("برمجة"))
        assert pattern.search(normalise("برمجة تطبيقات"))
        assert not pattern.search(normalise("البرمجةالمتقدمة"))


# ── Skill lookup ─────────────────────────────────────────────────────────────

class TestSkillLookup:

    def test_java_does_not_fire_inside_javascript(self):
        assert "java" not in find_skills("Frontend work in JavaScript and TypeScript")

    def test_skill_found_in_a_list(self):
        assert "python" in find_skills("Languages: Python, SQL, Java")


# ── End to end ───────────────────────────────────────────────────────────────

class TestMatchEndToEnd:

    def test_coverage_excludes_judgement_calls(self):
        """Personal qualities must not move the headline number.

        Otherwise coverage depends on how many soft skills a posting happened
        to list rather than on the candidate.
        """
        from src.Embedder import Embedder
        requirements = [
            {"text": "Experience with Python", "kind": "required", "category": "skill"},
            {"text": "Attention to detail", "kind": "required", "category": "skill"},
        ]
        resume = "Built data pipelines in Python and SQL for a retail analytics team."
        result = match(requirements, split_bullets(resume), Embedder(), resume)

        assert result["counts"][Verdict.JUDGEMENT] == 1
        assert result["decidable"] == 1
        assert result["coverage"] == 1.0

    def test_unsupported_rows_carry_no_evidence(self):
        """Showing a 'best match' under a verdict of 'no evidence' is a lie."""
        from src.Embedder import Embedder
        requirements = [{"text": "Experience with Kubernetes and Istio",
                         "kind": "required", "category": "skill"}]
        resume = "Wrote marketing copy and managed the content calendar."
        result = match(requirements, split_bullets(resume), Embedder(), resume)

        row = result["requirements"][0]
        assert row["status"] == Verdict.UNSUPPORTED
        assert row["evidence"] == ""
