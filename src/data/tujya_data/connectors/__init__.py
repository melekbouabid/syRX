"""Source connectors and their registry."""

from __future__ import annotations

from src.data.tujya_data.config import Settings
from src.data.tujya_data.connectors.base import Connector, Context, Link, SourceInfo
from src.data.tujya_data.connectors.ddinter import DDInterConnector
from src.data.tujya_data.connectors.openfda import OpenFdaLabelConnector
from src.data.tujya_data.connectors.pubmed import PubMedConnector

# The sources the Tujya module 2 corpus uses (the original synergy sources were removed,
# decision D85). Adding a source = writing one Connector subclass and listing it here.
_CONNECTORS: tuple[type[Connector], ...] = (
    DDInterConnector,
    PubMedConnector,
    OpenFdaLabelConnector,
)
REGISTRY: dict[str, type[Connector]] = {cls.info.name: cls for cls in _CONNECTORS}


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
