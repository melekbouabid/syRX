"""Pydantic row schemas, one per source table.

Every schema includes the five provenance fields, so a record without provenance cannot
pass validation.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Field(min_length=1, max_length=300)


class Record(BaseModel):
    """Base for all rows: provenance is mandatory."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    source: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=10)
    source_version: str = Field(min_length=1)
    license: str = Field(min_length=1)


# ------------------------------------------------------------- interactions


class InteractionRecord(Record):
    drug_a_name: str = Name
    drug_b_name: str = Name
    level: str | None = None
    drug_a_source_id: str | None = None
    drug_b_source_id: str | None = None


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


class LabelSectionRecord(Record):
    """One section of one openFDA drug label (Tujya RAG corpus); dosage is never stored."""

    ingredient: str = Name
    label_id: str = Field(min_length=1)
    set_id: str | None = None
    effective_time: str | None = None
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _no_dosage(self) -> Self:
        if "dosage" in self.section or "overdos" in self.section:
            raise ValueError("dosage sections are never stored")
        return self
