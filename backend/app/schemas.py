"""Request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    model_validator,
)

from app.models import Role

# Emails are compared case-insensitively, so they are stored in lower case.
Email = Annotated[EmailStr, AfterValidator(str.lower)]
Password = Annotated[str, StringConstraints(min_length=8, max_length=128)]
FullName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: Role
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


# --- sign in, sign up, own account ----------------------------------------------------------


class SignupIn(BaseModel):
    email: Email
    full_name: FullName
    password: Password


class LoginIn(BaseModel):
    # No format or length rule here: a wrong value is simply a failed sign-in.
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class ProfileUpdate(BaseModel):
    full_name: FullName


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=128)
    new_password: Password


# --- administration -------------------------------------------------------------------------


class AdminUserCreate(BaseModel):
    email: Email
    full_name: FullName
    password: Password
    role: Role = Role.USER
    is_active: bool = True


class AdminUserUpdate(BaseModel):
    email: Email | None = None
    full_name: FullName | None = None
    role: Role | None = None
    is_active: bool | None = None


class AdminPasswordReset(BaseModel):
    new_password: Password


class UserPage(BaseModel):
    items: list[UserOut]
    total: int
    page: int
    page_size: int


class AdminStats(BaseModel):
    users_total: int
    users_active: int
    users_inactive: int
    admins: int
    signups_last_7_days: int
    active_sessions: int


# --- Tujya: pills, scan, interactions -------------------------------------------------------
# Every answer about medicines carries its sources and the disclaimer (safety rules,
# docs/project_rules.md).

IngredientName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=2, max_length=100)
]
ClassId = Annotated[int, Field(ge=0, le=9999)]


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    title: str
    url: str
    license: str
    citation: str


class PillOut(BaseModel):
    """One identification class: a C3PI product (NDC9 + part) and its active ingredients."""

    class_id: int
    name: str
    generic_name: str
    labeled_by: str
    imprint: str
    color: str
    shape: str
    active_ingredients: list[str]


class PillList(BaseModel):
    items: list[PillOut]
    total: int
    sources: list[SourceOut]


class PillCandidate(BaseModel):
    class_id: int
    name: str
    generic_name: str
    active_ingredients: list[str]
    probability: float = Field(ge=0, le=1)


class PixelBox(BaseModel):
    """x1, y1 top-left and x2, y2 bottom-right, in pixels of the uploaded image."""

    x1: float
    y1: float
    x2: float
    y2: float


class NormalizedBox(BaseModel):
    """Same box divided by the image width (x) and height (y): values between 0 and 1."""

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class ScannedPill(BaseModel):
    box: PixelBox
    box_normalized: NormalizedBox
    detection_confidence: float = Field(ge=0, le=1)
    candidates: list[PillCandidate] = Field(description="Top-5 classes, most probable first")
    needs_confirmation: bool = Field(
        description="True when the top-1 probability is below confirm_threshold"
    )


class ScanTimings(BaseModel):
    detection_ms: float
    classification_ms: float
    total_ms: float = Field(description="Detection plus identification, ONNX Runtime")
    server_ms: float | None = Field(
        None, description="Whole request on the server (upload, decoding, database included)"
    )
    ocr_ms: float | None = Field(None, description="Reading of the printed text (boxes)")


class BoxIngredient(BaseModel):
    name: str = Field(description="Ingredient name of the interaction table (English)")
    method: Literal["brand", "printed_name", "rxnav"] = Field(
        description="brand table, ingredient name printed on the box, or RxNorm (RxNav)"
    )
    matched_text: str = Field(description="The text read on the box that gave this match")
    confidence: float = Field(ge=0, le=1, description="OCR confidence x word similarity")
    source: str
    reference: str
    url: str
    in_interaction_table: bool = Field(
        description="False: known ingredient, but absent from the interaction table"
    )


class BoxReading(BaseModel):
    """A medicine box: the text read and what it was linked to (never a guess)."""

    text: list[str] = Field(description="Lines read on the photo (at most 40)")
    brand: str | None = Field(None, description="Brand of config/brand_aliases.csv")
    brand_country: str | None = None
    brand_text: str | None = Field(None, description="The text read that matched the brand")
    ingredients: list[BoxIngredient]
    recognised: bool = Field(description="At least one active ingredient was found")
    confidence: float = Field(ge=0, le=1)
    rxnav: str = Field(description="Fallback status: enabled, or why it is disabled")


class ScanResult(BaseModel):
    kind: Literal["pill", "box", "unknown"] = Field(
        "pill",
        description="pill: pills detected; box: printed text of a medicine box; unknown: "
        "neither (refused, no identification)",
    )
    box: BoxReading | None = None
    image_width: int
    image_height: int
    inference_ms: float = Field(description="Detection plus identification, ONNX Runtime")
    timings: ScanTimings
    confirm_threshold: float
    pills: list[ScannedPill]
    message: str | None = None
    models: dict[str, Any]
    sources: list[SourceOut]
    disclaimer: str


class InteractionsIn(BaseModel):
    """At least two things to compare, pill classes (from a scan) and/or ingredient names."""

    pills: list[ClassId] = Field(default_factory=list, max_length=10)
    ingredients: list[IngredientName] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _two_to_ten(self) -> InteractionsIn:
        count = len(self.pills) + len(self.ingredients)
        if not 2 <= count <= 10:
            raise ValueError("give between 2 and 10 pills and ingredients in total")
        return self


class CheckedItem(BaseModel):
    kind: Literal["pill", "ingredient"]
    label: str
    class_id: int | None
    found: bool = Field(description="False when this database does not know the item")
    active_ingredients: list[str]


class InteractionOut(BaseModel):
    ingredient_a: str
    ingredient_b: str
    severity: str = Field(description="DDInter level: Major, Moderate, Minor or Unknown")
    description: str
    source: SourceOut


class PairOut(BaseModel):
    item_a: CheckedItem
    item_b: CheckedItem
    highest_severity: str | None
    interactions: list[InteractionOut]
    shared_ingredients: list[str] = Field(
        description="Active ingredients present in both items (same substance twice)"
    )
    message: str | None = Field(
        description="Set when nothing is listed: no known interaction, or nothing to check"
    )


class InteractionsResult(BaseModel):
    items: list[CheckedItem]
    pairs: list[PairOut]
    highest_severity: str | None
    message: str
    sources: list[SourceOut]
    disclaimer: str


class DemoImage(BaseModel):
    id: int
    class_id: int = Field(description="Class the photo shows (C3PI label), to compare with")
    expected_name: str
    why: str = Field(description="Why it is in the gallery: major, confirm, no-ingredient...")
    url: str
    thumbnail_url: str
    gradcam_url: str | None = Field(
        None, description="Precomputed Grad-CAM of this photo (D104), null when not computed"
    )


class DemoGallery(BaseModel):
    items: list[DemoImage]
    note: str


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=1000, description="French or English")
    pills: list[int] = Field(
        default_factory=list, max_length=10, description="Confirmed pill class_ids (scan)"
    )
    stream: bool = Field(False, description="Server-Sent Events instead of one JSON answer")


class CitationOut(BaseModel):
    n: int = Field(description="The number used in the answer: [n]")
    title: str
    url: str
    section: str
    snippet: str
    source_name: str
    doc_type: str = Field(description="label, abstract, interaction or brand")
    date: str
    cited: bool = Field(description="True if the answer cites [n]")
    doc_id: str = Field("", description="Document in the index (Documents page)")
    license: str = ""
    rerank_score: float | None = Field(
        None, description="Cross-encoder score when the search was reranked (D117)"
    )


class ToolCallOut(BaseModel):
    name: str
    args: dict[str, Any]
    chosen_by: str = Field(description="llm (tool calling) or rule (added by the graph)")
    ok: bool
    summary: str


class ChatResult(BaseModel):
    question: str
    answer: str = Field(description="Checked by the safety node; ends with the disclaimer")
    language: str
    ingredients: list[str] = Field(description="Active ingredients found in the question/pills")
    citations: list[CitationOut]
    tool_calls: list[ToolCallOut]
    safety: dict[str, Any] = Field(description="action ok / rewritten / refused, violations")
    model: str
    timings: dict[str, float]
    disclaimer: str


# ---------------------------------------------------------------- toxicity (bonus)


class ToxicityIn(BaseModel):
    molecules: list[str] = Field(
        min_length=1,
        max_length=2,
        description="One or two molecules: active ingredient names or SMILES",
        examples=[["warfarin", "fluconazole"]],
    )


class ToxPrediction(BaseModel):
    task: str = Field(description="Tox21 assay, e.g. SR-p53")
    assay: str
    probability: float = Field(description="Predicted probability of being active in the assay")
    test_roc_auc: float | None = Field(description="The model's ROC-AUC on this assay (test set)")


class ToxMolecule(BaseModel):
    query: str
    resolved_as: Literal["ingredient", "smiles"]
    ingredient: str | None
    smiles: str
    predictions: list[ToxPrediction]
    max_similarity_to_training: float
    in_domain: bool = Field(description="False: far from the training molecules")


class ToxError(BaseModel):
    query: str
    error: str


class ToxicityResult(BaseModel):
    wording: str = Field(description="Always: a model prediction on in vitro tests")
    molecules: list[ToxMolecule] = Field(description="Each molecule on its own")
    errors: list[ToxError]
    mixture_note: str
    low_probability_note: str
    model: dict[str, Any]
    source: dict[str, str]
    disclaimer: str


# --- Endpoints of the course guide (GET /health, GET /model/metadata, POST /predict) --------


class Strict(BaseModel):
    """Response bodies of the guide endpoints: no undeclared field, values checked."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


Task = Literal["detect", "classify"]


class LoadedModel(Strict):
    """One ONNX model as published in models.json (src/models/publish.py)."""

    task: Task
    file: str
    registered_model: str | None = Field(None, description="MLflow Model Registry name")
    version: int | None = Field(None, description="Registry version served")
    aliases: list[str] = Field(default_factory=list, description="e.g. champion, served")
    precision: str = Field("fp32", description="fp32, fp16, int8-static or int8-dynamic")
    variant_of: int | None = Field(None, description="Champion version this variant comes from")
    run_id: str | None = None
    sha256: str | None = None


class GpuStatus(Strict):
    available: bool = Field(description="ONNX Runtime can use a CUDA GPU")
    available_providers: list[str]
    active_providers: list[str] = Field(description="Providers of the loaded sessions")


class HealthOut(Strict):
    status: Literal["ok", "degraded"] = Field(description="degraded: models not loaded")
    model_loaded: bool
    detail: str | None = None
    api_version: str
    gpu: GpuStatus
    models: list[LoadedModel]
    published_at: str | None = None


class ModelMetadata(Strict):
    task: Task
    name: str = Field(description="Registry name, or the file name without the Registry")
    architecture: str = Field(description="e.g. yolov8n, yolo11s-cls")
    version: int | None
    aliases: list[str]
    precision: str
    variant_of: int | None
    run_id: str | None
    input_size: int = Field(description="Square input, pixels")
    metrics: dict[str, float] = Field(description="Validation and test scores, latency")
    classes: list[str] = Field(description="Output index -> class (C3PI folder / class_id)")
    num_classes: int


class MetadataOut(Strict):
    detector: ModelMetadata
    classifier: ModelMetadata
    published_at: str | None = None
    disclaimer: str


class ClassScore(Strict):
    class_id: int
    probability: float = Field(ge=0, le=1)


class PredictedPill(Strict):
    box: PixelBox
    box_normalized: NormalizedBox
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    top5: list[ClassScore] = Field(description="Most probable classes first")


class PredictTimings(Strict):
    detection_ms: float
    classification_ms: float
    inference_ms: float = Field(description="Detection plus identification, ONNX Runtime")
    server_ms: float = Field(description="Whole request on the server, decoding included")


class PredictOut(Strict):
    image_width: int
    image_height: int
    count: int
    pills: list[PredictedPill]
    timings: PredictTimings
    model_versions: dict[str, int | None] = Field(description="detect / classify versions")
    disclaimer: str


# ------------------------------------------------------------------ indexed documents (D114)


class IndexInfoOut(BaseModel):
    store: str = Field(description="qdrant (default), pgvector or sqlite")
    collection: str = Field(description="Qdrant collection (or SQL table) of the chunks")
    points: int = Field(description="Chunks in the store")
    embedding_model: str
    dim: int
    indexed_at: str
    documents: int
    chunks: int
    payload_indexes: list[str] = Field(description="Indexed metadata fields (filters)")
    dashboard_url: str | None = Field(description="Qdrant web dashboard, opened on the host")


class DocumentOut(BaseModel):
    doc_id: str = Field(description="openfda:<set id>, pubmed:<PMID> or ddinter:<id>:<id>")
    doc_type: str = Field(description="label, abstract or interaction")
    title: str
    source_name: str
    url: str
    date: str
    version: str
    license: str
    language: str
    article_type: str | None
    is_retracted: bool
    ingredients: list[str] = Field(description="Active ingredients the document is about")
    mentioned_ingredients: list[str] = Field(description="Other ingredients it mentions")
    sections: list[str]
    chunk_count: int


class DocumentsPage(BaseModel):
    total: int = Field(description="Documents matching the filters")
    offset: int
    limit: int
    items: list[DocumentOut]
    facets: dict[str, dict[str, int]] = Field(
        description="Matching documents per doc_type, language and source"
    )
    index: IndexInfoOut


class DocumentChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    section: str
    title: str
    text: str = Field(description="Chunk text; sentences with a dose are shown as [...]")
    content_hash: str = Field(description="SHA-256 of text + metadata + embedding model")


class DocumentDetail(DocumentOut):
    chunks: list[DocumentChunkOut]


# --- Data drift of the scanned photos (GET /monitoring/drift, step 2.8) ----------------------

DriftStatus = Literal["unavailable", "insufficient", "ok", "warning", "drift"]


class DriftFeature(Strict):
    """One statistic of the window against the reference photos."""

    feature: str
    description: str
    reference_n: int = Field(description="Reference photos with this statistic")
    current_n: int = Field(description="Photos of the window with this statistic")
    reference_mean: float | None
    current_mean: float | None
    ks_statistic: float | None = Field(description="Two-sample Kolmogorov-Smirnov statistic")
    ks_pvalue: float | None
    psi: float | None = Field(description="Population Stability Index, bins = reference deciles")
    status: Literal["insufficient", "ok", "warning", "drift"]


class DriftReport(Strict):
    status: DriftStatus = Field(
        description="unavailable: no reference file; insufficient: fewer photos than "
        "min_samples; otherwise the worst status of the statistics"
    )
    window_size: int = Field(description="Photos in the sliding window (memory only)")
    window_max: int
    photos_seen: int = Field(description="Photos scanned since the API started")
    drifted: list[str] = Field(description="Statistics in drift")
    features: list[DriftFeature]
    thresholds: dict[str, float] = Field(description="params.yaml `drift`, from the reference")
    reference: dict[str, Any] = Field(description="Photos and served models of the reference")
