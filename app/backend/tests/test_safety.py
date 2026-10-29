"""Agent building blocks without any model: drug names in questions, and the safety rules."""

from __future__ import annotations

from app.medicines import DISCLAIMER, NO_KNOWN_INTERACTION
from app.rag.names import NameIndex
from app.rag.safety import SentenceFilter, check_answer, is_dose_question, sentence_problems


def test_names_in_french_and_english() -> None:
    index = NameIndex(["warfarin", "fluconazole", "acetaminophen", "vitamin b12", "folic acid"])
    assert index.find("La warfarine et le Fluconazole, ensemble ?") == ["warfarin", "fluconazole"]
    assert index.find("Paracétamol et vitamine B12") == ["acetaminophen", "vitamin b12"]
    assert index.find("acide folique") == ["folic acid"]
    assert index.lookup("WARFARINE") == "warfarin" and index.lookup("aspirin") is None
    assert index.find("Est-ce que je peux prendre ça ?") == []


def test_dose_and_safe_sentences() -> None:
    assert sentence_problems("Prenez 50 mg le matin [1].") == ["dose"]
    assert sentence_problems("Prendre deux fois par jour.") == ["dose"]
    assert sentence_problems("Cette association est sûre.") == ["safe"]
    assert sentence_problems("This combination is safe.") == ["safe"]
    assert sentence_problems("Voir la notice : sur la notice [1].") == []
    assert sentence_problems("Tujya a identifié un comprimé [1].") == []
    assert sentence_problems("Espacez les prises de 2 heures.") == ["uncited_timing"]
    assert sentence_problems("Espacez les prises de 2 heures [1].") == []
    assert is_dose_question("Quelle dose de paracétamol puis-je prendre ?")
    assert is_dose_question("How much ibuprofen can I take?")
    assert not is_dose_question("Le fluconazole interagit-il avec la warfarine ?")


def test_check_answer_rewrites_and_refuses() -> None:
    report = check_answer(
        "Le fluconazole augmente l'INR [1]. Cette association est sans danger. Voir [9].",
        2,
        interaction_found=True,
    )
    assert report.action == "rewritten"
    assert "sans danger" not in report.answer and "[9]" not in report.answer
    assert report.answer.endswith(DISCLAIMER) and report.cited == [1]
    assert set(report.violations) == {"safe", "bad_citation"}

    no_interaction = check_answer(
        "Ces deux médicaments sont sûrs ensemble [1].", 1, interaction_found=False
    )
    assert NO_KNOWN_INTERACTION in no_interaction.answer and "sûrs" not in no_interaction.answer

    uncited = check_answer("Le losartan peut donner des vertiges.", 2, interaction_found=False)
    assert "no_citation" in uncited.violations and "Sources consultées : [1], [2]" in uncited.answer

    nothing = check_answer("Peu importe [1].", 0, interaction_found=False)
    assert nothing.action == "refused" and nothing.answer.endswith(DISCLAIMER)

    clean = check_answer(
        f"Le losartan peut donner des vertiges [1].\n\n{DISCLAIMER}", 1, interaction_found=False
    )
    assert clean.action == "ok" and clean.answer.count(DISCLAIMER) == 1


def test_sentence_filter_holds_back_unsafe_sentences() -> None:
    stream = SentenceFilter()
    shown: list[str] = []
    for piece in ["Le fluco", "nazole est sûr. Il aug", "mente l'INR [1]. Prenez 5", "0 mg."]:
        shown += stream.feed(piece)
    shown += stream.flush()
    assert "".join(shown).strip() == "Il augmente l'INR [1]."


def test_lab_values_are_not_doses_and_missing_citations_are_attributed() -> None:
    assert sentence_problems("Contraindicated when eGFR is below 30 mL/min/1.73 m2 [1].") == []
    assert sentence_problems("Glucose above 200 mg/dL [1].") == []
    assert sentence_problems("Take 500 mg [1].") == ["dose"]
    label = (
        "Metformin label: Contraindications. Metformin is contraindicated in patients with "
        "acute or chronic metabolic acidosis, including diabetic ketoacidosis."
    )
    report = check_answer(
        "Metformin is contraindicated in acute or chronic metabolic acidosis, including "
        "diabetic ketoacidosis [see Contraindications (4)]. [2]",
        2,
        interaction_found=False,
        source_texts=["Warfarin and bleeding.", label],
    )
    assert "see Contraindications" not in report.answer
    assert report.cited == [2] and report.answer.count("[2]") == 1
    added = check_answer(
        "Metformin is contraindicated in acute or chronic metabolic acidosis.",
        2,
        interaction_found=False,
        source_texts=["Warfarin and bleeding.", label],
    )
    assert added.cited == [2] and added.violations == ["citation_added"] and added.action == "ok"
    unrelated = check_answer(
        "Losartan causes dizziness in elderly people.",
        2,
        interaction_found=False,
        source_texts=["Warfarin and bleeding.", label],
    )
    assert unrelated.cited == [] and "no_citation" in unrelated.violations


def test_attribution_by_meaning_cites_the_closest_source_only_when_clear() -> None:
    import asyncio

    from app.rag.embeddings import FakeEmbedder
    from app.rag.safety import attribute_by_meaning

    sources = [
        "Warfarin label: bleeding risk and prothrombin time monitoring with azole antifungals.",
        "Metformin label: contraindicated in severe renal impairment and metabolic acidosis.",
    ]
    text = "Metformin is contraindicated in severe renal impairment and metabolic acidosis."
    cited, added = asyncio.run(attribute_by_meaning(text, sources, FakeEmbedder(), threshold=0.5))
    assert added == 1 and cited.endswith("acidosis [2].")
    already = "Warfarin needs monitoring [1]."
    assert asyncio.run(attribute_by_meaning(already, sources, FakeEmbedder(), threshold=0.5)) == (
        already,
        0,
    )
    unrelated = "Losartan may cause dizziness in some elderly people."
    _, none = asyncio.run(attribute_by_meaning(unrelated, sources, FakeEmbedder(), threshold=0.9))
    assert none == 0


def test_dosing_advice_without_a_number_is_a_dose_sentence() -> None:
    for sentence in (
        "Un ajustement de la dose de warfarine peut être nécessaire [1].",
        "Il faut peut-être ajuster la dose [2].",
        "Dosage adjustment may be needed [1].",
        "Reduce the dose of simvastatin [1].",
        "A dose reduction is recommended [3].",
    ):
        assert sentence_problems(sentence) == ["dose"], sentence
    assert sentence_problems("Le fluconazole augmente le temps de prothrombine [1].") == []


def test_snippets_hide_dose_sentences() -> None:
    from app.rag.safety import redact_doses

    text = (
        "Fluconazole increases warfarin exposure. Administration of 40 mg/day citalopram did "
        "not change it. The tolvaptan dose should be reduced. Monitor the INR."
    )
    assert redact_doses(text) == "Fluconazole increases warfarin exposure. […] Monitor the INR."


def test_non_toxicity_claims_are_removed_but_facts_about_toxicity_stay() -> None:
    claims = [
        "La warfarine n'est pas toxique [1].",
        "La warfarine n'est pas identifiée comme toxique par le modèle [1].",
        "Il n'y a pas de toxicité [1].",
        "Aucune toxicité notable n'est prédite [1].",
        "The model indicates no significant toxicity [1].",
        "Fluconazole is not predicted to be toxic [1].",
        "Low probability of toxicity in every assay [1].",
    ]
    for sentence in claims:
        assert "not_toxic_claim" in sentence_problems(sentence), sentence
    for sentence in (
        "Fluconazole is hepatotoxic in rare cases [1].",
        "La toxicité hépatique est rare [1].",
        "The probability of activity in SR-MMP is 0.83 [1].",
    ):
        assert sentence_problems(sentence) == [], sentence


def test_sentence_filter_can_hold_back_more_sentences() -> None:
    held = SentenceFilter(hold=lambda sentence: "Tox21" in sentence)
    shown = held.feed("La warfarine est un anticoagulant [1]. Le modèle Tox21 dit ")
    shown += held.feed("autre chose [2]. Fin [1].")
    shown += held.flush()
    assert "".join(shown).strip() == "La warfarine est un anticoagulant [1]. Fin [1]."
