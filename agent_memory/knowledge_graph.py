"""Knowledge graph — typed entities and semantic relations over memory.

Unlike the similarity-based :class:`~agent_memory.graph.MemoryGraph`, this
graph stores **named entities** (people, organisations, locations, concepts)
and typed **relationships** between them, extracted from memory content.

The result is a queryable knowledge base that can answer questions like:
  - Who works at Acme Corp?
  - What APIs does the project use?
  - Which memories mention "Karan"?

Usage::

    from agent_memory import Memory
    from agent_memory.knowledge_graph import KnowledgeGraph, GraphBuilder

    memory = Memory(persist_dir=".agent_memory")

    # Build with the default composite extractor
    kg = GraphBuilder().build(memory.store)

    # Query
    entity = kg.find_entity("Karan")
    for rel in kg.relations_for(entity.id):
        print(rel.relation, "→", rel.target)

    # Path between entities
    path = kg.path("Karan", "Acme Corp")
"""
from __future__ import annotations

import heapq
import re
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry
from agent_memory.store import MemoryStore

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rich entity types
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    # People & roles
    PERSON        = "person"
    ROLE          = "role"             # job title, position
    # Organisations
    ORGANIZATION  = "organization"
    TEAM          = "team"
    DEPARTMENT    = "department"
    # Places
    LOCATION      = "location"
    COUNTRY       = "country"
    CITY          = "city"
    REGION        = "region"
    # Technology — languages, frameworks, tools
    TECHNOLOGY    = "technology"
    LANGUAGE      = "language"         # programming language
    FRAMEWORK     = "framework"
    LIBRARY       = "library"
    TOOL          = "tool"
    # Infrastructure & services
    API           = "api"
    SERVICE       = "service"          # SaaS, microservice
    DATABASE      = "database"
    SERVER        = "server"
    ENDPOINT      = "endpoint"
    ENVIRONMENT   = "environment"      # dev / staging / production
    CONTAINER     = "container"        # Docker image, k8s pod
    CLUSTER       = "cluster"
    # Products & software
    PRODUCT       = "product"
    VERSION       = "version"
    REPOSITORY    = "repository"
    BRANCH        = "branch"
    # Projects & work
    PROJECT       = "project"
    FEATURE       = "feature"
    BUG           = "bug"
    TICKET        = "ticket"           # GitHub issue, Jira ticket
    PR            = "pull_request"
    # Security
    CREDENTIAL    = "credential"       # API key, token
    PERMISSION    = "permission"
    CERTIFICATE   = "certificate"
    # Configuration
    CONFIGURATION = "configuration"
    SETTING       = "setting"
    METRIC        = "metric"           # KPI, measurement
    # Time
    EVENT         = "event"
    DATE          = "date"
    DEADLINE      = "deadline"
    RELEASE       = "release"
    # Knowledge
    CONCEPT       = "concept"
    FACT          = "fact"
    PREFERENCE    = "preference"
    POLICY        = "policy"
    RULE          = "rule"
    # Contacts / communication
    EMAIL         = "email"
    URL           = "url"
    CONTACT       = "contact"
    MESSAGE       = "message"
    # Misc
    QUANTITY      = "quantity"
    COMMAND       = "command"          # CLI command, script
    FILE          = "file"
    UNKNOWN       = "unknown"


# ---------------------------------------------------------------------------
# Rich relation types
# ---------------------------------------------------------------------------


class RelationType(str, Enum):
    # Work & organisation
    WORKS_AT          = "works_at"
    MANAGES           = "manages"
    REPORTS_TO        = "reports_to"
    OWNS              = "owns"
    COLLABORATES_WITH = "collaborates_with"
    PART_OF           = "part_of"
    MEMBER_OF         = "member_of"
    FOUNDED           = "founded"
    ASSIGNED_TO       = "assigned_to"
    # Place
    LOCATED_IN        = "located_in"
    HEADQUARTERED_IN  = "headquartered_in"
    DEPLOYED_IN       = "deployed_in"      # service deployed to environment
    # Technology — structural
    USES              = "uses"
    DEPENDS_ON        = "depends_on"
    EXTENDS           = "extends"
    IMPLEMENTS        = "implements"
    REPLACES          = "replaces"
    COMPATIBLE_WITH   = "compatible_with"
    DEPRECATED_BY     = "deprecated_by"
    INTEGRATES_WITH   = "integrates_with"
    CALLS             = "calls"            # service A calls service B
    # Technology — versioning
    HAS_VERSION       = "has_version"
    UPGRADED_TO       = "upgraded_to"
    FORKED_FROM       = "forked_from"
    # Authorship / ownership
    CREATED_BY        = "created_by"
    AUTHORED_BY       = "authored_by"
    MAINTAINED_BY     = "maintained_by"
    REVIEWED_BY       = "reviewed_by"
    APPROVED_BY       = "approved_by"
    # Hierarchy / containment
    IS_A              = "is_a"
    HAS               = "has"
    CONTAINS          = "contains"
    HAS_ROLE          = "has_role"
    CONFIGURES        = "configures"
    # Security
    AUTHENTICATES     = "authenticates"
    AUTHORIZES        = "authorizes"
    GRANTS            = "grants"
    REVOKES           = "revokes"
    # Knowledge / reference
    KNOWS             = "knows"
    RELATED_TO        = "related_to"
    MENTIONED_IN      = "mentioned_in"
    PREFERS           = "prefers"
    REQUIRES          = "requires"
    REFERENCES        = "references"
    OVERRIDES         = "overrides"
    # Lifecycle / time
    SCHEDULED_FOR     = "scheduled_for"
    EXPIRED_ON        = "expired_on"
    PUBLISHED_AT      = "published_at"
    TRIGGERS          = "triggers"
    RESULTS_IN        = "results_in"
    # Misc
    TAGGED_WITH       = "tagged_with"
    LINKED_TO         = "linked_to"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeEntity:
    """A named entity in the knowledge graph."""

    id: str
    name: str
    type: EntityType
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    source_memory_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def add_alias(self, alias: str) -> None:
        if alias and alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)

    def all_names(self) -> list[str]:
        return [self.name] + self.aliases


@dataclass
class KnowledgeRelation:
    """A directed, typed relationship."""

    id: str
    source_id: str
    relation: str           # RelationType value or free-form
    target: str             # entity id when target_is_entity else raw value
    target_is_entity: bool = False
    confidence: float = 1.0
    source_memory_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction result (transport object between strategy and graph)
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    entities: list[KnowledgeEntity] = field(default_factory=list)
    relations: list[KnowledgeRelation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction strategy interface  (Dependency Inversion + Open/Closed)
# ---------------------------------------------------------------------------


class EntityExtractionStrategy(ABC):
    """Abstract strategy for extracting entities and relations from text.

    Implement this to add a new extraction backend (LLM-based, spaCy, etc.)
    without modifying ``KnowledgeGraph`` or ``GraphBuilder``.
    """

    @abstractmethod
    def extract(self, text: str, memory_id: str) -> ExtractionResult:
        """Extract entities and relations from *text* that originated from *memory_id*."""


# ---------------------------------------------------------------------------
# Pattern-based extraction strategy
# ---------------------------------------------------------------------------

_REL_PATTERNS: list[tuple[re.Pattern[str], str, EntityType, EntityType]] = [
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+works?\s+(?:at|for)\s+([A-Z][a-zA-Z&\s]+?)(?:\.|,|$)", re.I),
     RelationType.WORKS_AT, EntityType.PERSON, EntityType.ORGANIZATION),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+manages?\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.MANAGES, EntityType.PERSON, EntityType.TEAM),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+(?:is )?(?:based|located)\s+in\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$)", re.I),
     RelationType.LOCATED_IN, EntityType.PERSON, EntityType.LOCATION),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+(?:created|built|developed|founded)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.CREATED_BY, EntityType.PRODUCT, EntityType.PERSON),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+uses?\s+([A-Z][a-zA-Z\s+.#]+?)(?:\.|,|$)", re.I),
     RelationType.USES, EntityType.PERSON, EntityType.TECHNOLOGY),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+(?:depends on|requires)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.DEPENDS_ON, EntityType.PRODUCT, EntityType.TECHNOLOGY),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+is part of\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.PART_OF, EntityType.UNKNOWN, EntityType.ORGANIZATION),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+(?:replaces|deprecates)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.REPLACES, EntityType.TECHNOLOGY, EntityType.TECHNOLOGY),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+extends?\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.EXTENDS, EntityType.TECHNOLOGY, EntityType.TECHNOLOGY),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+knows?\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     RelationType.KNOWS, EntityType.PERSON, EntityType.PERSON),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+prefers?\s+([A-Z][a-zA-Z\s+.#]+?)(?:\.|,|$)", re.I),
     RelationType.PREFERS, EntityType.PERSON, EntityType.TECHNOLOGY),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+(?:is headquartered|headquartered)\s+in\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$)", re.I),
     RelationType.HEADQUARTERED_IN, EntityType.ORGANIZATION, EntityType.LOCATION),
]

_ENTITY_PATTERNS: list[tuple[re.Pattern[str], EntityType]] = [
    (re.compile(r"\bmy name is\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)", re.I), EntityType.PERSON),
    (re.compile(r"\bI(?:'m| am)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)", re.I), EntityType.PERSON),
    (re.compile(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s+is a[n]?\s+(?:company|organisation|organization|startup|firm|corp)", re.I), EntityType.ORGANIZATION),
    (re.compile(r"(?:located|based|headquartered)\s+in\s+([A-Z][a-zA-Z]+(?:,?\s[A-Z][a-zA-Z]+)*)", re.I), EntityType.LOCATION),
    (re.compile(r"\b(Python|JavaScript|TypeScript|Go|Rust|Java|C\+\+|Ruby|PHP|Swift|Kotlin)\b"), EntityType.TECHNOLOGY),
    (re.compile(r"\b(FastAPI|Django|Flask|React|Vue|Angular|Next\.js|Express|Spring)\b"), EntityType.TECHNOLOGY),
    (re.compile(r"\b(PostgreSQL|MySQL|SQLite|MongoDB|Redis|DynamoDB|Cassandra)\b"), EntityType.TECHNOLOGY),
    (re.compile(r"\b(AWS|GCP|Azure|Docker|Kubernetes|Terraform|GitHub|GitLab)\b"), EntityType.SERVICE),
    (re.compile(r"\bversion\s+([\d]+\.[\d]+(?:\.[\d]+)?)\b", re.I), EntityType.VERSION),
    (re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\s+(?:Corp|Inc|Ltd|LLC|GmbH)\b"), EntityType.ORGANIZATION),
    (re.compile(r"\bdeadline(?:\s+is)?\s+(.+?)(?:\.|,|$)", re.I), EntityType.DEADLINE),
    (re.compile(r"\b(https?://[^\s,]+)", re.I), EntityType.URL),
]


def _make_entity(name: str, etype: EntityType, memory_id: str) -> KnowledgeEntity:
    return KnowledgeEntity(
        id=str(uuid.uuid4()),
        name=name.strip(),
        type=etype,
        source_memory_ids=[memory_id] if memory_id else [],
        confidence=0.78,
    )


def _make_relation(src_id: str, rel: str, tgt_id: str, memory_id: str, *, target_is_entity: bool = True) -> KnowledgeRelation:
    return KnowledgeRelation(
        id=str(uuid.uuid4()),
        source_id=src_id,
        relation=rel,
        target=tgt_id,
        target_is_entity=target_is_entity,
        confidence=0.80,
        source_memory_id=memory_id,
    )


class PatternExtractionStrategy(EntityExtractionStrategy):
    """Regex-based entity and relation extraction — zero external dependencies."""

    def extract(self, text: str, memory_id: str) -> ExtractionResult:
        result = ExtractionResult()
        entity_map: dict[str, KnowledgeEntity] = {}   # name.lower() → entity

        def _get_or_create(name: str, etype: EntityType) -> KnowledgeEntity:
            key = name.strip().lower()
            if key not in entity_map:
                entity_map[key] = _make_entity(name.strip(), etype, memory_id)
                result.entities.append(entity_map[key])
            else:
                if entity_map[key].type == EntityType.UNKNOWN and etype != EntityType.UNKNOWN:
                    entity_map[key].type = etype
            return entity_map[key]

        # Named-entity patterns
        for pattern, etype in _ENTITY_PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1).strip().rstrip(".,;")
                if len(name) >= 2:
                    _get_or_create(name, etype)

        # Relation patterns
        for pattern, rel_type, src_type, tgt_type in _REL_PATTERNS:
            for m in pattern.finditer(text):
                src_name = m.group(1).strip()
                tgt_name = m.group(2).strip().rstrip(".,;")
                if len(src_name) < 2 or len(tgt_name) < 2:
                    continue
                src = _get_or_create(src_name, src_type)
                tgt = _get_or_create(tgt_name, tgt_type)
                result.relations.append(
                    _make_relation(src.id, rel_type, tgt.id, memory_id)
                )

        # Capitalised multi-word phrases (fallback)
        cap_re = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")
        for m in cap_re.finditer(text):
            name = m.group(1)
            if name.lower() not in entity_map and len(name) >= 4:
                _get_or_create(name, EntityType.UNKNOWN)

        # Mention relations (entity appears in this memory)
        for entity in result.entities:
            result.relations.append(_make_relation(
                entity.id, RelationType.MENTIONED_IN, memory_id,
                memory_id, target_is_entity=False
            ))

        return result


class SpacyExtractionStrategy(EntityExtractionStrategy):
    """spaCy NER extraction — auto-detected; falls back to empty result if not installed."""

    _LABEL_MAP: dict[str, EntityType] = {
        "PERSON": EntityType.PERSON,
        "ORG":    EntityType.ORGANIZATION,
        "GPE":    EntityType.LOCATION,
        "LOC":    EntityType.LOCATION,
        "PRODUCT": EntityType.PRODUCT,
        "EVENT":  EntityType.EVENT,
        "DATE":   EntityType.DATE,
        "MONEY":  EntityType.QUANTITY,
        "PERCENT": EntityType.QUANTITY,
    }

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self._nlp: Any = None
        try:
            import spacy
            self._nlp = spacy.load(model)
        except (ImportError, OSError):
            pass

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def extract(self, text: str, memory_id: str) -> ExtractionResult:
        if self._nlp is None:
            return ExtractionResult()
        result = ExtractionResult()
        doc = self._nlp(text)
        for ent in doc.ents:
            etype = self._LABEL_MAP.get(ent.label_, EntityType.UNKNOWN)
            e = _make_entity(ent.text, etype, memory_id)
            e.attributes["ner_label"] = ent.label_
            e.confidence = 0.82
            result.entities.append(e)
            result.relations.append(_make_relation(
                e.id, RelationType.MENTIONED_IN, memory_id, memory_id, target_is_entity=False
            ))
        return result


class CompositeExtractionStrategy(EntityExtractionStrategy):
    """Runs multiple strategies and merges their results.

    Entities with the same name (case-insensitive) are merged; relations are
    combined.  The ordering of strategies determines precedence for type
    assignment: later strategies can upgrade UNKNOWN types but not override.
    """

    def __init__(self, strategies: list[EntityExtractionStrategy]) -> None:
        self._strategies = strategies

    def extract(self, text: str, memory_id: str) -> ExtractionResult:
        merged = ExtractionResult()
        name_map: dict[str, KnowledgeEntity] = {}

        for strategy in self._strategies:
            sub = strategy.extract(text, memory_id)
            for entity in sub.entities:
                key = entity.name.lower()
                if key not in name_map:
                    name_map[key] = entity
                    merged.entities.append(entity)
                else:
                    existing = name_map[key]
                    if existing.type == EntityType.UNKNOWN and entity.type != EntityType.UNKNOWN:
                        existing.type = entity.type
                    existing.attributes.update(entity.attributes)
            for rel in sub.relations:
                merged.relations.append(rel)

        return merged


# ---------------------------------------------------------------------------
# Graph builder  (SRP: builds; does not query)
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Builds a ``KnowledgeGraph`` from a memory store.

    Depends on ``EntityExtractionStrategy`` (abstract), not any concrete
    extractor — satisfying the Dependency Inversion Principle.
    """

    def __init__(
        self,
        strategy: EntityExtractionStrategy | None = None,
    ) -> None:
        if strategy is None:
            # Default: pattern + spaCy (if available)
            pattern_strategy = PatternExtractionStrategy()
            spacy_strategy = SpacyExtractionStrategy()
            if spacy_strategy.available:
                strategy = CompositeExtractionStrategy([pattern_strategy, spacy_strategy])
            else:
                strategy = pattern_strategy
        self._strategy = strategy

    def build(self, store: MemoryStore) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        entries = store.list_all(limit=10_000)
        for entry in entries:
            self.ingest_entry(entry, kg)
        return kg

    def ingest_entry(self, entry: MemoryEntry, kg: KnowledgeGraph) -> None:
        text = f"{entry.query} {entry.response}"
        result = self._strategy.extract(text, entry.id)

        for entity in result.entities:
            kg._merge_entity(entity)

        for rel in result.relations:
            # Remap entity IDs that were merged
            rel.source_id = kg._canonical_id(rel.source_id)
            if rel.target_is_entity:
                rel.target = kg._canonical_id(rel.target)
            kg._add_relation(rel)

        # Tag-based concept entities
        for tag in entry.tags:
            if len(tag) >= 3:
                tag_entity = kg.add_entity(tag, EntityType.CONCEPT, source_memory_id=entry.id, confidence=0.70)
                # Link memory's first extracted entity to this tag concept
                for eid in list(kg.entities.keys())[:3]:
                    if eid != tag_entity.id:
                        kg.add_relation(
                            eid, RelationType.TAGGED_WITH, tag_entity.id,
                            target_is_entity=True, confidence=0.65, source_memory_id=entry.id,
                        )


# ---------------------------------------------------------------------------
# KnowledgeGraph  (SRP: stores + queries only)
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Typed entity + relation graph.

    Responsibilities (SRP): data storage, query, merge, export.
    Does NOT know how entities are extracted — that is ``GraphBuilder``'s job.

    Build via ``GraphBuilder().build(store)`` or ``memory.knowledge_graph()``.
    """

    def __init__(self) -> None:
        self.entities: dict[str, KnowledgeEntity] = {}
        self.relations: list[KnowledgeRelation] = []
        self._by_name: dict[str, str] = {}                         # lower name/alias → entity_id
        self._adj: dict[str, list[str]] = defaultdict(list)        # entity_id → [relation_id list]
        self._rel_index: dict[str, KnowledgeRelation] = {}         # relation_id → KnowledgeRelation (O(1) lookup)
        self._id_map: dict[str, str] = {}                          # old_id → canonical_id (after merges)

    # ------------------------------------------------------------------
    # Convenience factory (for memory.knowledge_graph())
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        store: MemoryStore,
        strategy: EntityExtractionStrategy | None = None,
    ) -> KnowledgeGraph:
        """Shortcut: ``GraphBuilder(strategy).build(store)``."""
        return GraphBuilder(strategy).build(store)

    # ------------------------------------------------------------------
    # Mutation (public write interface)
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.UNKNOWN,
        attributes: dict[str, Any] | None = None,
        *,
        source_memory_id: str = "",
        confidence: float = 1.0,
    ) -> KnowledgeEntity:
        """Add or merge an entity by name. Returns the (possibly merged) entity."""
        existing = self.find_entity(name)
        if existing:
            if attributes:
                existing.attributes.update(attributes)
            if source_memory_id and source_memory_id not in existing.source_memory_ids:
                existing.source_memory_ids.append(source_memory_id)
            if existing.type == EntityType.UNKNOWN and entity_type != EntityType.UNKNOWN:
                existing.type = entity_type
            return existing
        eid = str(uuid.uuid4())
        entity = KnowledgeEntity(
            id=eid, name=name, type=entity_type,
            attributes=attributes or {},
            source_memory_ids=[source_memory_id] if source_memory_id else [],
            confidence=confidence,
        )
        self.entities[eid] = entity
        self._by_name[name.lower()] = eid
        return entity

    def add_relation(
        self,
        source_id: str,
        relation: str,
        target: str,
        *,
        target_is_entity: bool = False,
        confidence: float = 1.0,
        source_memory_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeRelation:
        rid = str(uuid.uuid4())
        rel = KnowledgeRelation(
            id=rid, source_id=source_id, relation=relation,
            target=target, target_is_entity=target_is_entity,
            confidence=confidence, source_memory_id=source_memory_id,
            metadata=metadata or {},
        )
        self.relations.append(rel)
        self._rel_index[rid] = rel          # O(1) index
        self._adj[source_id].append(rid)
        if target_is_entity:
            self._adj[target].append(rid)
        return rel

    def merge_entities(self, keep_id: str, remove_id: str) -> KnowledgeEntity:
        """Merge *remove_id* into *keep_id*, re-pointing all relations."""
        keep = self.entities.get(keep_id)
        remove = self.entities.get(remove_id)
        if not keep or not remove:
            raise ValueError("One or both entity IDs not found")
        keep.add_alias(remove.name)
        for alias in remove.aliases:
            keep.add_alias(alias)
        keep.attributes.update(remove.attributes)
        keep.source_memory_ids.extend(
            m for m in remove.source_memory_ids if m not in keep.source_memory_ids
        )
        for rel in self.relations:
            if rel.source_id == remove_id:
                rel.source_id = keep_id
            if rel.target_is_entity and rel.target == remove_id:
                rel.target = keep_id
        del self.entities[remove_id]
        self._by_name[remove.name.lower()] = keep_id
        for alias in remove.aliases:
            self._by_name[alias.lower()] = keep_id
        self._id_map[remove_id] = keep_id
        return keep

    # ------------------------------------------------------------------
    # Query interface (read-only operations)
    # ------------------------------------------------------------------

    def find_entity(self, name: str) -> KnowledgeEntity | None:
        eid = self._by_name.get(name.lower())
        return self.entities.get(eid) if eid else None

    def relations_for(
        self, entity_id: str, *, relation_type: str | None = None
    ) -> list[KnowledgeRelation]:
        """Return relations for *entity_id* in O(degree) time using the index."""
        rels = [
            self._rel_index[rid]
            for rid in self._adj.get(entity_id, [])
            if rid in self._rel_index
        ]
        if relation_type:
            rels = [r for r in rels if r.relation == relation_type]
        return rels

    def neighbors(self, entity_id: str) -> list[tuple[KnowledgeEntity, str]]:
        result: list[tuple[KnowledgeEntity, str]] = []
        for rel in self.relations_for(entity_id):
            if rel.target_is_entity:
                other_id = rel.target if rel.source_id == entity_id else rel.source_id
                other = self.entities.get(other_id)
                if other:
                    result.append((other, rel.relation))
        return result

    def find_by_type(self, entity_type: EntityType) -> list[KnowledgeEntity]:
        return [e for e in self.entities.values() if e.type == entity_type]

    def search(self, query: str) -> list[KnowledgeEntity]:
        """Text search over names, aliases, and attributes."""
        q = query.lower()
        return [
            e for e in self.entities.values()
            if (q in e.name.lower()
                or any(q in a.lower() for a in e.aliases)
                or any(q in str(v).lower() for v in e.attributes.values()))
        ]

    def search_prefix(self, prefix: str) -> list[KnowledgeEntity]:
        """Find entities whose name or alias starts with *prefix* (case-insensitive).

        Iterates the sorted ``_by_name`` keys so the scan short-circuits once
        the prefix is no longer matched — O(k + log n) amortised where k is
        the number of matches.
        """
        p = prefix.lower()
        results: set[str] = set()
        for key, eid in sorted(self._by_name.items()):
            if key.startswith(p):
                results.add(eid)
            elif key > p and results:
                break   # sorted order means no more matches
        return [self.entities[eid] for eid in results if eid in self.entities]

    def path(self, source_name: str, target_name: str) -> list[KnowledgeEntity]:
        """BFS shortest hop-count path between two entities.

        For a confidence-weighted shortest path use :meth:`shortest_path`.
        """
        src = self.find_entity(source_name)
        tgt = self.find_entity(target_name)
        if not src or not tgt:
            return []
        if src.id == tgt.id:
            return [src]
        visited: set[str] = set()
        prev: dict[str, str | None] = {src.id: None}
        queue: deque[str] = deque([src.id])
        while queue:
            node_id = queue.popleft()
            if node_id == tgt.id:
                # Reconstruct path from prev map — O(path_length)
                path: list[str] = []
                cur: str | None = tgt.id
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return [self.entities[eid] for eid in path if eid in self.entities]
            if node_id in visited:
                continue
            visited.add(node_id)
            for rel in self.relations_for(node_id):
                if rel.target_is_entity:
                    nxt = rel.target if rel.source_id == node_id else rel.source_id
                    if nxt not in visited and nxt not in prev:
                        prev[nxt] = node_id
                        queue.append(nxt)
        return []

    def shortest_path(
        self, source_name: str, target_name: str
    ) -> list[KnowledgeEntity]:
        """Dijkstra's shortest path weighted by relation confidence.

        Edge cost = ``1 - relation.confidence``, so high-confidence edges
        are preferred.  Falls back to an empty list when no path exists.

        Time complexity: O((V + E) log V) — better than BFS when edges have
        varying confidence and the optimal path is not the hop-shortest one.
        """
        src = self.find_entity(source_name)
        tgt = self.find_entity(target_name)
        if not src or not tgt:
            return []
        if src.id == tgt.id:
            return [src]

        INF = float("inf")
        dist: dict[str, float] = defaultdict(lambda: INF)
        dist[src.id] = 0.0
        prev: dict[str, str | None] = {src.id: None}
        # Min-heap: (cost, entity_id)
        heap: list[tuple[float, str]] = [(0.0, src.id)]

        while heap:
            d, uid = heapq.heappop(heap)
            if d > dist[uid]:
                continue           # stale entry — skip
            if uid == tgt.id:
                break
            for rel in self.relations_for(uid):
                if not rel.target_is_entity:
                    continue
                nxt = rel.target if rel.source_id == uid else rel.source_id
                cost = 1.0 - rel.confidence      # lower = better
                new_d = d + cost
                if new_d < dist[nxt]:
                    dist[nxt] = new_d
                    prev[nxt] = uid
                    heapq.heappush(heap, (new_d, nxt))

        if tgt.id not in prev:
            return []

        path: list[str] = []
        cur: str | None = tgt.id
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return [self.entities[eid] for eid in path if eid in self.entities]

    def clusters(self, *, min_confidence: float = 0.0) -> list[list[KnowledgeEntity]]:
        """Return connected components using Union-Find (DSU).

        **DSA note:** Union-Find with path-compression and union-by-rank
        achieves near-O(1) amortised per operation — O(α(n)) where α is the
        inverse Ackermann function, practically constant.  This replaces the
        earlier DFS/BFS approach which rebuilds the component structure from
        scratch every call.

        Only edges with ``relation.confidence >= min_confidence`` are
        considered when grouping.
        """
        node_ids = list(self.entities.keys())
        if not node_ids:
            return []

        parent = {nid: nid for nid in node_ids}
        rank: dict[str, int] = {nid: 0 for nid in node_ids}

        def find(x: str) -> str:
            # Path-compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        for rel in self.relations:
            if not rel.target_is_entity or rel.confidence < min_confidence:
                continue
            if rel.source_id in parent and rel.target in parent:
                union(rel.source_id, rel.target)

        component_map: dict[str, list[KnowledgeEntity]] = defaultdict(list)
        for nid in node_ids:
            root = find(nid)
            component_map[root].append(self.entities[nid])

        return sorted(component_map.values(), key=len, reverse=True)

    def memories_for_entity(
        self, entity_id: str, store: MemoryStore | None = None
    ) -> list[MemoryEntry]:
        entity = self.entities.get(entity_id)
        if not entity or store is None:
            return []
        mem_ids: set[str] = set(entity.source_memory_ids)
        for rel in self.relations_for(entity_id):
            if rel.source_memory_id:
                mem_ids.add(rel.source_memory_id)
        return [e for mid in mem_ids if (e := store.get(mid)) is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [
                {"id": e.id, "name": e.name, "type": e.type.value,
                 "aliases": e.aliases, "attributes": e.attributes, "confidence": e.confidence}
                for e in self.entities.values()
            ],
            "relations": [
                {"source": r.source_id, "relation": r.relation, "target": r.target,
                 "target_is_entity": r.target_is_entity, "confidence": r.confidence}
                for r in self.relations
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers used by GraphBuilder
    # ------------------------------------------------------------------

    def _canonical_id(self, entity_id: str) -> str:
        return self._id_map.get(entity_id, entity_id)

    def _merge_entity(self, entity: KnowledgeEntity) -> KnowledgeEntity:
        """Merge an incoming entity into the graph (deduplicating by name)."""
        existing = self.find_entity(entity.name)
        if existing:
            if entity.type != EntityType.UNKNOWN and existing.type == EntityType.UNKNOWN:
                existing.type = entity.type
            existing.attributes.update(entity.attributes)
            for mid in entity.source_memory_ids:
                if mid not in existing.source_memory_ids:
                    existing.source_memory_ids.append(mid)
            self._id_map[entity.id] = existing.id
            return existing
        self.entities[entity.id] = entity
        self._by_name[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self._by_name[alias.lower()] = entity.id
        self._id_map[entity.id] = entity.id
        return entity

    def _add_relation(self, rel: KnowledgeRelation) -> None:
        self.relations.append(rel)
        self._rel_index[rel.id] = rel        # O(1) lookup
        self._adj[rel.source_id].append(rel.id)
        if rel.target_is_entity:
            self._adj[rel.target].append(rel.id)
