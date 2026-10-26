"""Source connectors and their registry."""

from __future__ import annotations

from synergyrx_data.config import Settings
from synergyrx_data.connectors.base import Connector, Context, Link, SourceInfo
from synergyrx_data.connectors.chembl import ChemblConnector
from synergyrx_data.connectors.ddinter import DDInterConnector
from synergyrx_data.connectors.depmap import DepMapConnector
from synergyrx_data.connectors.drugbank_manual import DrugBankManualConnector
from synergyrx_data.connectors.drugcomb import DrugCombConnector
from synergyrx_data.connectors.europepmc import EuropePmcConnector
from synergyrx_data.connectors.gdsc import GdscConnector
from synergyrx_data.connectors.nci_almanac import NciAlmanacConnector
from synergyrx_data.connectors.oneil import ONeilConnector
from synergyrx_data.connectors.pubchem import PubChemConnector
from synergyrx_data.connectors.pubmed import PubMedConnector
from synergyrx_data.connectors.pubmed_baseline import PubMedBaselineConnector
from synergyrx_data.connectors.pubtator import PubTatorConnector
from synergyrx_data.connectors.tdc import TdcConnector

# Adding a source = writing one Connector subclass and listing it here.
_CONNECTORS: tuple[type[Connector], ...] = (
    DrugCombConnector,
    TdcConnector,
    ONeilConnector,
    NciAlmanacConnector,
    DepMapConnector,
    GdscConnector,
    DDInterConnector,
    DrugBankManualConnector,
    PubChemConnector,
    ChemblConnector,
    PubMedConnector,
    PubMedBaselineConnector,
    EuropePmcConnector,
    PubTatorConnector,
)
REGISTRY: dict[str, type[Connector]] = {cls.info.name: cls for cls in _CONNECTORS}

SYNERGY_SOURCES = tuple(n for n, c in REGISTRY.items() if c.info.category == "synergy")


def source_names() -> list[str]:
    """All source names, in the order they run (by stage, then registration)."""
    return sorted(REGISTRY, key=lambda n: (REGISTRY[n].info.stage, list(REGISTRY).index(n)))


def get_connector(name: str, ctx: Context) -> Connector:
    if name not in REGISTRY:
        raise KeyError(f"unknown source {name!r}; known sources: {', '.join(source_names())}")
    return REGISTRY[name](ctx)


def is_enabled(name: str, settings: Settings) -> bool:
    config = settings.sources.get(name)
    return config.enabled if config else REGISTRY[name].info.default_enabled


def allowed_hosts() -> set[str]:
    """Every host a connector declares. The HTTP client refuses all others."""
    return {host for cls in REGISTRY.values() for host in cls.info.hosts}


def all_links() -> list[tuple[str, Link]]:
    """(source name, link) for every URL listed in the brief, without duplicates."""
    seen: set[str] = set()
    links = []
    for name in REGISTRY:
        for link in REGISTRY[name].info.links:
            if link.url not in seen:
                seen.add(link.url)
                links.append((name, link))
    return links


__all__ = [
    "REGISTRY",
    "SYNERGY_SOURCES",
    "Connector",
    "Context",
    "Link",
    "SourceInfo",
    "all_links",
    "allowed_hosts",
    "get_connector",
    "is_enabled",
    "source_names",
]
