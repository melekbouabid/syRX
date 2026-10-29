"""Therapeutics Data Commons `DrugSyn` datasets.

The file is always downloaded by the polite client, from the Harvard Dataverse address
PyTDC itself uses, into the folder PyTDC reads from. It is then loaded through
`tdc.multi_pred.DrugSyn` when PyTDC is installed, and through a restricted loader otherwise
(PyTDC pins numpy<2 / pandas<3 and does not install on Python 3.13+).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._safe_pickle import load_dataframe_pickle
from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.connectors.base import (
    Artifact,
    Connector,
    Context,
    Link,
    ManualStepRequired,
    PlannedAction,
    SourceInfo,
)
from synergyrx_data.connectors.schemas import Record, TdcRecord

_REQUIRED = ("Drug1_ID", "Drug2_ID", "Cell_Line_ID", "Drug1", "Drug2")
_SCORES = {
    "Synergy_Bliss": "synergy_bliss",
    "Synergy_Loewe": "synergy_loewe",
    "Synergy_HSA": "synergy_hsa",
    "Synergy_ZIP": "synergy_zip",
    "CSS": "css",
}


class TdcConnector(Connector):
    info = SourceInfo(
        name="tdc",
        title="Therapeutics Data Commons DrugSyn",
        category="synergy",
        access="package",
        homepage="https://tdcommons.ai/multi_pred_tasks/drugsyn/",
        license="CC-BY-4.0 (TDC lists the dataset licence as 'Not Specified. CC BY 4.0')",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms_url="https://tdcommons.ai/multi_pred_tasks/drugsyn/",
        citation=(
            "Huang K, et al. Therapeutics Data Commons: machine learning datasets and tasks "
            "for drug discovery and development. NeurIPS Datasets and Benchmarks 2021. "
            "arXiv:2102.09548"
        ),
        hosts=("dataverse.harvard.edu", "tdcommons.ai", "arxiv.org"),
        links=(
            Link(
                "https://arxiv.org/pdf/2102.09548",
                "Therapeutics Data Commons paper",
                "html",
                "checked",
            ),
            Link("https://tdcommons.ai/multi_pred_tasks/drugsyn/", "TDC DrugSyn task", "html", ""),
        ),
        notes=(
            "The `drugcomb` set is the NCI-60 (ALMANAC) part of DrugComb, so it overlaps the "
            "DrugComb connector. `oncopolypharmacology` is the O'Neil screen (1.6 GB file)."
        ),
    )
    default_options: ClassVar[dict[str, Any]] = {
        "dataset": "drugcomb",
        "loader": "auto",  # auto | pytdc | builtin
        "sample_seed": 0,
        # File ids and sizes as published in PyTDC 1.1.15 (tdc/metadata.py).
        "files": {
            "drugcomb": {"file_id": 4215720, "size_bytes": 36_394_069},
            "oncopolypharmacology": {"file_id": 4167358, "size_bytes": 1_621_643_413},
        },
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"synergy": TdcRecord}

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.dataset = str(self.options["dataset"]).lower()
        if self.dataset not in self.options["files"]:
            raise ValueError(f"unknown TDC DrugSyn dataset {self.dataset!r}")
        self._file = self.options["files"][self.dataset]
        if self.version == self.default_version:
            self.version = f"dataverse-{self._file['file_id']}"

    @property
    def url(self) -> str:
        return f"https://dataverse.harvard.edu/api/access/datafile/{self._file['file_id']}"

    @property
    def filename(self) -> str:
        return f"{self.dataset}.pkl"  # the name PyTDC looks for

    async def discover(self) -> list[PlannedAction]:
        plan = self.plan_file(self.url, self.filename, self._file["size_bytes"])
        note = "pickle cannot be read partially: the whole file is fetched, then sampled"
        return [replace(plan, note=note if self.sample else "")]

    async def fetch(self) -> list[Artifact]:
        # The Dataverse Data Access API is a documented API, hence kind="api".
        return [
            await self.fetch_file(
                self.url, self.filename, expected_size=self._file["size_bytes"], kind="api"
            )
        ]

    def _load(self, path: Path) -> pd.DataFrame:
        loader = self.options["loader"]
        self.stats["loader"] = "builtin"
        if loader in ("auto", "pytdc"):
            try:
                from tdc.multi_pred import DrugSyn
            except ImportError as exc:
                if loader == "pytdc":
                    raise ManualStepRequired(
                        "loader=pytdc but PyTDC is not installed. Install the extra with "
                        "`pip install 'synergyrx-data[tdc]'` on Python 3.11 or 3.12."
                    ) from exc
            else:
                frame = DrugSyn(name=self.dataset, path=str(path.parent)).get_data()
                if all(column in frame.columns for column in _REQUIRED):
                    self.stats["loader"] = "pytdc"
                    return frame
                self.log.warning("tdc.pytdc_reduced_table", extra={"columns": list(frame.columns)})
        return load_dataframe_pickle(path)

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        artifact = artifacts[0]
        frame = self._load(artifact.path)
        missing = [column for column in _REQUIRED if column not in frame.columns]
        if missing:
            raise ValueError(f"TDC table lacks expected columns: {missing}")
        self.stats["raw_rows"] = len(frame)
        if self.sample and self.sample < len(frame):
            frame = frame.sample(n=self.sample, random_state=self.options["sample_seed"])
            frame = frame.sort_index()
        scores = dict(_SCORES)
        if self.dataset == "oncopolypharmacology":
            scores = {"Y": "synergy_loewe"}  # TDC's label for this set is Loewe synergy
        table = pd.DataFrame(
            {
                "drug_a_name": frame["Drug1_ID"],
                "drug_b_name": frame["Drug2_ID"],
                "cell_line_name": frame["Cell_Line_ID"],
                "drug_a_smiles": frame["Drug1"],
                "drug_b_smiles": frame["Drug2"],
                "study_id": f"TDC:{self.dataset}",
                **{target: frame[src] for src, target in scores.items() if src in frame.columns},
            }
        )
        return {"synergy": self.provenance(artifact).apply(synergy_frame(table))}
