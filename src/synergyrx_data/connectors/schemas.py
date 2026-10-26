"""Pydantic row schemas, one per source table.

Every schema includes the five provenance fields, so a record without provenance cannot
pass validation.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Name = Field(min_length=1, max_length=300)


class Record(BaseModel):
    """Base for all rows: provenance is mandatory."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    source: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=10)
    source_version: str = Field(min_length=1)
    license: str = Field(min_length=1)


# ----------------------------------------------------------------- synergy


class SynergyRecord(Record):
    """One measured drug pair on one cell line, as the source reports it."""

    drug_a_name: str = Name
    drug_b_name: str = Name
    cell_line_name: str = Name
    tissue: str | None = None
    study_id: str | None = None
    source_record_id: str | None = None
    drug_a_source_id: str | None = None
    drug_b_source_id: str | None = None
    drug_a_smiles: str | None = None
    drug_b_smiles: str | None = None
    synergy_bliss: FiniteFloat | None = None
    synergy_loewe: FiniteFloat | None = None
    synergy_hsa: FiniteFloat | None = None
    synergy_zip: FiniteFloat | None = None
    css: FiniteFloat | None = None
    combo_score: FiniteFloat | None = None

    @model_validator(mode="after")
    def _has_a_score(self) -> Self:
        scores = (
            self.synergy_bliss,
            self.synergy_loewe,
            self.synergy_hsa,
            self.synergy_zip,
            self.css,
            self.combo_score,
        )
        if all(value is None for value in scores):
            raise ValueError("row has no synergy score at all")
        return self


class DrugCombRecord(SynergyRecord):
    source_record_id: str = Field(min_length=1)  # block_id


class ONeilRecord(SynergyRecord):
    synergy_loewe: FiniteFloat  # the only score this dataset publishes
    drug_a_smiles: str = Field(min_length=1)
    drug_b_smiles: str = Field(min_length=1)


class TdcRecord(SynergyRecord):
    drug_a_smiles: str = Field(min_length=1)
    drug_b_smiles: str = Field(min_length=1)


class AlmanacRecord(SynergyRecord):
    combo_score: FiniteFloat
    drug_a_source_id: str = Field(pattern=r"^NSC-\d+$")
    drug_b_source_id: str = Field(pattern=r"^NSC-\d+$")
    study_id: str = Field(min_length=1)


# -------------------------------------------------------------- cell lines


class DepMapModelRecord(Record):
    depmap_id: str = Field(pattern=r"^ACH-\d{6}$")
    cell_line_name: str | None = None
    stripped_name: str | None = None
    ccle_name: str | None = None
    lineage: str | None = None
    primary_disease: str | None = None


class SangerModelRecord(Record):
    sanger_model_id: str = Field(pattern=r"^SIDM\d+$")
    cell_line_name: str = Name
    depmap_id: str | None = None
    tissue: str | None = None
    cancer_type: str | None = None


# --------------------------------------------------------------- chemistry


class PubChemRecord(Record):
    query_name: str = Name
    status: str = Field(pattern=r"^(found|ambiguous|not_found)$")
    n_cids: int = Field(ge=0)
    pubchem_cid: int | None = Field(default=None, gt=0)
    smiles: str | None = None
    connectivity_smiles: str | None = None
    inchikey: str | None = Field(default=None, pattern=r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
    title: str | None = None
    synonyms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _found_has_structure(self) -> Self:
        if self.status != "not_found" and self.pubchem_cid is None:
            raise ValueError("a found compound must have a CID")
        return self


class ChemblRecord(Record):
    query: str = Field(min_length=1)
    query_type: str = Field(pattern=r"^(inchikey|name)$")
    chembl_id: str = Field(pattern=r"^CHEMBL\d+$")
    pref_name: str | None = None
    max_phase: float | None = None
    inchikey: str | None = None


class InteractionRecord(Record):
    drug_a_name: str = Name
    drug_b_name: str = Name
    level: str | None = None
    drug_a_source_id: str | None = None
    drug_b_source_id: str | None = None


class DrugVocabularyRecord(Record):
    drugbank_id: str = Field(pattern=r"^DB\d{5}$")
    name: str = Name
    synonyms: list[str] = Field(default_factory=list)
    inchikey: str | None = None
    cas: str | None = None
    unii: str | None = None


# -------------------------------------------------------------- literature


class ArticleRecord(Record):
    pmid: str = Field(pattern=r"^\d+$")
    title: str | None = None
    abstract: str | None = None
    journal: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    doi: str | None = None
    pmcid: str | None = None
    language: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    article_type: str = "research"
    is_retracted: bool = False
    is_retraction_notice: bool = False
    has_erratum: bool = False
    has_expression_of_concern: bool = False
    has_abstract: bool = False


class EuropePmcRecord(Record):
    pmid: str = Field(pattern=r"^\d+$")
    pmcid: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    is_open_access: bool = False
    in_epmc: bool = False
    has_pdf: bool = False
    article_license: str | None = None


class AnnotationRecord(Record):
    pmid: str = Field(pattern=r"^\d+$")
    entity_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    identifier: str | None = None
    section: str | None = None
    start: int = Field(ge=0)
    length: int = Field(ge=0)
