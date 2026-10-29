"""O'Neil et al. 2016 combination screen, as released with DeepSynergy."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import ONeilRecord, Record


class ONeilConnector(Connector):
    info = SourceInfo(
        name="oneil",
        title="O'Neil 2016 combination screen (DeepSynergy release)",
        category="synergy",
        access="bulk",
        homepage="https://bioinf.jku.at/software/DeepSynergy/",
        license="Unspecified (no licence stated on the download page)",
        terms_url="https://bioinf.jku.at/software/DeepSynergy/",
        citation=(
            "O'Neil J, et al. An unbiased oncology compound screen to identify novel "
            "combination strategies. Mol Cancer Ther. 2016;15(6):1155-62. "
            "doi:10.1158/1535-7163.MCT-15-0843; Preuer K, et al. DeepSynergy. "
            "Bioinformatics. 2018;34(9):1538-46. doi:10.1093/bioinformatics/btx806"
        ),
        hosts=("bioinf.jku.at",),
        links=(
            Link(
                "https://bioinf.jku.at/software/DeepSynergy",
                "DeepSynergy page (O'Neil dataset and code)",
                "html",
                "checked",
            ),
        ),
        redistribution=(
            "No licence is published: treat as research use only, cite both papers and do "
            "not redistribute."
        ),
        notes="Publishes Loewe synergy only. Each row is already averaged over replicates.",
    )
    default_version = "2017-12"
    default_options: ClassVar[dict[str, Any]] = {
        "base_url": "https://bioinf.jku.at/software/DeepSynergy/",
        "labels": "labels.csv",
        "labels_size_bytes": 1_570_785,
        "smiles": "smiles.csv",
        "smiles_size_bytes": 2_507,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"synergy": ONeilRecord}

    def _url(self, key: str) -> str:
        return f"{self.options['base_url']}{self.options[key]}"

    async def discover(self) -> list[PlannedAction]:
        return [
            self.plan_file(
                self._url("labels"),
                self.options["labels"],
                self.options["labels_size_bytes"],
                sample_rows=self.sample,
            ),
            self.plan_file(
                self._url("smiles"), self.options["smiles"], self.options["smiles_size_bytes"]
            ),
        ]

    async def fetch(self) -> list[Artifact]:
        labels = await self.fetch_file(
            self._url("labels"), self.options["labels"], sample_rows=self.sample
        )
        smiles = await self.fetch_file(self._url("smiles"), self.options["smiles"])
        return [labels, smiles]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        labels_artifact, smiles_artifact = artifacts
        labels = pd.read_csv(labels_artifact.path, dtype={"cell_line": "string"})
        labels = labels.rename(columns={labels.columns[0]: "source_record_id"})
        smiles = pd.read_csv(
            smiles_artifact.path, header=None, names=["name", "smiles"], dtype="string"
        )
        lookup = dict(zip(smiles["name"].str.strip(), smiles["smiles"].str.strip(), strict=True))
        table = synergy_frame(
            labels.assign(
                cell_line_name=labels["cell_line"],
                synergy_loewe=labels["synergy"],
                study_id="ONEIL",
                drug_a_smiles=labels["drug_a_name"].map(lookup),
                drug_b_smiles=labels["drug_b_name"].map(lookup),
            )
        )
        self.stats = {"raw_rows": len(labels), "drugs_with_smiles": len(lookup)}
        return {"synergy": self.provenance(labels_artifact).apply(table)}
