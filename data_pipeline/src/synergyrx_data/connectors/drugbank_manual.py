"""DrugBank (licence-gated, disabled by default).

This connector never touches the network. It only reads a file that a licensed user has
downloaded from DrugBank themselves and placed under `data/manual/drugbank/`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors.base import (
    Artifact,
    Connector,
    Link,
    ManualStepRequired,
    PlannedAction,
    SourceInfo,
)
from synergyrx_data.connectors.schemas import DrugVocabularyRecord, Record

_COLUMNS = {
    "drugbank_id": "DrugBank ID",
    "name": "Common name",
    "cas": "CAS",
    "unii": "UNII",
    "inchikey": "Standard InChI Key",
}


class DrugBankManualConnector(Connector):
    info = SourceInfo(
        name="drugbank",
        title="DrugBank (manual file only)",
        category="chemistry",
        access="manual",
        homepage="https://go.drugbank.com/",
        license=(
            "DrugBank licence: an account and an accepted licence are required; terms depend "
            "on the file (the open vocabulary is CC0, the full database is not)"
        ),
        terms_url="https://go.drugbank.com/",
        citation=(
            "Knox C, et al. DrugBank 6.0: the DrugBank Knowledgebase for 2024. "
            "Nucleic Acids Res. 2024;52(D1):D1265-75. doi:10.1093/nar/gkad976"
        ),
        hosts=("go.drugbank.com",),
        links=(Link("https://go.drugbank.com/", "DrugBank", "html", "verify"),),
        optional=True,
        default_enabled=False,
        redistribution="Never redistribute. Keep the file outside version control.",
        notes="Reads the `drugbank vocabulary.csv` export. Nothing is ever downloaded.",
    )
    default_version = "user-supplied"
    default_options: ClassVar[dict[str, Any]] = {"filename": "drugbank vocabulary.csv"}
    schemas: ClassVar[dict[str, type[Record]]] = {"vocabulary": DrugVocabularyRecord}

    def _instructions(self) -> str:
        target = self.rel(self.manual_dir / self.options["filename"])
        return (
            "DrugBank is licence-gated and is never downloaded automatically. Sign in at "
            "https://go.drugbank.com/, accept the licence that fits your use, download the "
            f"vocabulary CSV yourself and save it as {target}."
        )

    async def discover(self) -> list[PlannedAction]:
        path = self.manual_dir / self.options["filename"]
        note = "read the manually placed file" if path.is_file() else self._instructions()
        return [PlannedAction(self.name, "manual", self.rel(path), None, None, note)]

    async def fetch(self) -> list[Artifact]:
        artifact = self.manual_artifact(self.options["filename"], self.info.homepage)
        if artifact is None:
            raise ManualStepRequired(self._instructions())
        return [artifact]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        raw = pd.read_csv(artifacts[0].path, dtype="string", nrows=self.sample)
        vocabulary = pd.DataFrame(
            {target: raw[source].str.strip() for target, source in _COLUMNS.items()}
        )
        synonyms = raw["Synonyms"].fillna("")
        vocabulary["synonyms"] = [
            [s.strip() for s in value.split("|") if s.strip()] for value in synonyms
        ]
        return {"vocabulary": self.provenance(artifacts[0]).apply(vocabulary)}
