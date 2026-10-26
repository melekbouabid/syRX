"""DDInter 2.0 drug-drug interactions (optional, disabled by default).

DDInter is licensed CC BY-NC-SA 4.0: non-commercial and share-alike. The connector refuses
to download anything until someone has read the terms and set `terms_acknowledged: true`.
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
from synergyrx_data.connectors.schemas import InteractionRecord, Record

TERMS = "https://ddinter2.scbdd.com/terms/"


class DDInterConnector(Connector):
    info = SourceInfo(
        name="ddinter",
        title="DDInter 2.0 drug-drug interactions",
        category="interaction",
        access="bulk",
        homepage="https://ddinter2.scbdd.com/",
        license="CC-BY-NC-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        terms_url=TERMS,
        citation=(
            "Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded "
            "data coverage, new interaction types, and improved user interface. "
            "Nucleic Acids Res. 2025;53(D1):D1356-62. doi:10.1093/nar/gkae726"
        ),
        hosts=("ddinter2.scbdd.com",),
        links=(
            Link("https://ddinter2.scbdd.com/", "DDInter", "html", "checked"),
            Link(TERMS, "DDInter terms and conditions", "html", ""),
        ),
        optional=True,
        default_enabled=False,
        redistribution=(
            "Non-commercial and share-alike. Personal, informational or scholarly use only; "
            "never ship it in a commercial product and do not redistribute."
        ),
        notes="Clinical interaction severity (Major / Moderate / Minor), not synergy.",
    )
    default_version = "2.0"
    default_options: ClassVar[dict[str, Any]] = {
        "terms_acknowledged": False,
        "base_url": "https://ddinter2.scbdd.com/static/media/download/",
        "codes": ["A", "B", "D", "H", "L", "P", "R", "V"],
        "max_age_days": 90,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"interactions": InteractionRecord}

    def _files(self) -> list[tuple[str, str]]:
        names = [f"ddinter_downloads_code_{code}.csv" for code in self.options["codes"]]
        return [(self.options["base_url"] + name, name) for name in names]

    async def discover(self) -> list[PlannedAction]:
        if not self.options["terms_acknowledged"]:
            return [
                PlannedAction(
                    self.name,
                    "manual",
                    TERMS,
                    None,
                    None,
                    "read the terms, then set sources.ddinter.options.terms_acknowledged: true",
                )
            ]
        return [
            self.plan_file(url, name, None, sample_rows=self.sample) for url, name in self._files()
        ]

    async def fetch(self) -> list[Artifact]:
        if not self.options["terms_acknowledged"]:
            raise ManualStepRequired(
                f"DDInter is CC BY-NC-SA 4.0 (non-commercial, share-alike). Read {TERMS} and, "
                "if your use complies, set sources.ddinter.options.terms_acknowledged: true "
                "in config.yaml."
            )
        return [
            await self.fetch_file(
                url,
                name,
                sample_rows=self.sample,
                max_age_days=self.options["max_age_days"],
            )
            for url, name in self._files()
        ]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        frames = []
        for artifact in artifacts:
            raw = pd.read_csv(artifact.path, dtype="string")
            table = pd.DataFrame(
                {
                    "drug_a_name": raw["Drug_A"].str.strip(),
                    "drug_b_name": raw["Drug_B"].str.strip(),
                    "level": raw["Level"].str.strip(),
                    "drug_a_source_id": raw["DDInterID_A"],
                    "drug_b_source_id": raw["DDInterID_B"],
                }
            )
            frames.append(self.provenance(artifact).apply(table))
        if not frames:
            return {}
        # The per-category files overlap: the same pair is listed under each drug's category.
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(
            ["drug_a_source_id", "drug_b_source_id", "level"]
        )
        return {"interactions": merged.reset_index(drop=True)}
