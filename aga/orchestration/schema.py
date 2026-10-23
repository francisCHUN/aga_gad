
from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class DatasetStats(BaseModel):

    model_config = ConfigDict(extra="allow")

    dataset: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    num_attr_anomalies: int
    num_stru_anomalies: int
    num_total_anomalies: int
    avg_degree: float
    degree_std: float
    homophily: float = Field(..., description="edge-wise label agreement fraction")
    num_communities: int
    class_distribution: Dict[str, int] = Field(
        default_factory=dict, description="topic/class label -> node count"
    )
    stats_scope: str = Field(
        "train_only",
        description="which node set the label-dependent stats cover: 'train_only' or 'full_graph'",
    )


class AnomalyCoverage(BaseModel):
    model_config = ConfigDict(extra="allow")

    family: str
    score: Optional[float] = None
    status: Literal["observed", "unmeasured"] = "observed"
    note: str = ""


class AnomalyGeometry(BaseModel):
    model_config = ConfigDict(extra="allow")

    stru_degree_mean: Optional[float] = None
    stru_degree_std: Optional[float] = None
    stru_degree_median: Optional[float] = None
    stru_degree_p90: Optional[float] = None
    normal_ctx_cos_mean: Optional[float] = None
    attr_ctx_cos_mean: Optional[float] = None
    normal_feat_norm_mean: Optional[float] = None
    anomaly_feat_norm_mean: Optional[float] = None
    attr_feat_norm_mean: Optional[float] = None


class DatasetReport(BaseModel):

    stats: DatasetStats
    coverage: List[AnomalyCoverage] = Field(
        default_factory=list,
        description="Coverage bars, one per anomaly family, 0 meaning absent.",
    )
    coverage_bars: Dict[str, str] = Field(default_factory=dict)
    anomaly_geometry: Optional[AnomalyGeometry] = None
    summary: str = Field("", description="one-paragraph natural-language description")


class MissingPattern(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    severity: float = Field(..., ge=0.0, le=1.0, description="how badly the dataset needs it")
    rationale: str
    proposed: bool = Field(
        False,
        description="True if name is NOT in the fixed-core typology and needs a builder later.",
    )
    proposed_spec: Optional[str] = Field(
        None, description="if proposed, a textual spec the user can implement as a builder later"
    )
    gap_id: str = Field("", description="unique id threading review + generation, e.g. 'gap-001'")
    claim: str = Field(
        "", description="the concrete gap statement to verify (what is claimed missing)"
    )
    scope: str = Field(
        "train_all",
        description="verification scope: 'train_all' or a specific community/range description",
    )
    pattern_definition: str = Field(
        "", description="what makes a node count as matching this pattern (verifiable criteria)"
    )
    evidence_needed: List[str] = Field(
        default_factory=list,
        description="kinds of evidence required to verify the claim",
    )
    initial_rationale: str = Field(
        "", description="original reasoning behind the gap hypothesis (kept for audit)"
    )


class MissingPatterns(BaseModel):

    existing_patterns: List[str]
    missing: List[MissingPattern]
    global_note: str = ""


class SampleCard(BaseModel):

    sample_id: int
    gap_id: str
    group: Literal["targeted", "stratified", "boundary"]
    reason: str = ""
    stratum: str = ""
    community: int = -1
    degree: int = 0
    cross_community_ratio: float = 0.0
    ctx_cos: Optional[float] = None
    degree_percentile: Optional[float] = None
    ctx_cos_percentile: Optional[float] = None
    cross_ratio_percentile: Optional[float] = None
    text: str = ""
    text_truncated: Optional[bool] = Field(
        None, description="True when the card text was cut (full text stays in the csv)"
    )
    neighbour_ids: List[int] = Field(default_factory=list, description="representative neighbours")
    neighbour_texts: List[str] = Field(default_factory=list)
    neighbour_selection: str = ""
    label: Optional[int] = None
    label_source: str = "none"
    limitations: List[str] = Field(default_factory=list)


class SampleFeedback(BaseModel):

    sample_id: int
    gap_id: str
    verdict: Literal["matches", "not_matches", "undecidable"] = "undecidable"
    comment: str = ""
    source: Literal["human", "llm"] = "human"


class GapFeedback(BaseModel):

    gap_id: str
    review_status: Literal[
        "pending",
        "confirmed",
        "revised_confirmed",
        "rejected",
        "insufficient_evidence",
    ] = "pending"
    revised_claim: Optional[str] = None
    reviewed_scope: Optional[str] = None
    human_comment: str = ""
    request_resample: bool = False
    source: Literal["human", "llm"] = "human"
    coverage_criterion: Optional[str] = None


class ConditionAssessment(BaseModel):

    condition_index: int = Field(..., ge=0)
    status: Literal["supported", "contradicted", "unknown"]
    note: str = ""


class SampleJudgement(BaseModel):

    sample_id: int
    assessments: List[ConditionAssessment] = Field(default_factory=list)
    reason: str = ""


class SampleJudgements(BaseModel):

    judgements: List[SampleJudgement] = Field(default_factory=list)


class GapSuggestion(BaseModel):

    gap_id: str
    suggested_status: Literal[
        "pending", "confirmed", "revised_confirmed", "rejected",
        "insufficient_evidence",
    ]
    evidence_summary: str = ""
    reason: str = ""


class GapSuggestions(BaseModel):

    suggestions: List[GapSuggestion] = Field(default_factory=list)


class GapReview(BaseModel):

    gap_id: str
    pattern_name: str
    review_status: Literal[
        "pending", "confirmed", "revised_confirmed", "rejected", "insufficient_evidence"
    ] = "pending"
    supporting_sample_ids: List[int] = Field(default_factory=list)
    counterexample_sample_ids: List[int] = Field(default_factory=list)
    undecidable_sample_ids: List[int] = Field(default_factory=list)
    revised_claim: Optional[str] = None
    reviewed_scope: Optional[str] = None
    evidence_summary: str = ""
    limitations: List[str] = Field(default_factory=list)
    human_comment: str = ""
    coverage_criterion: Optional[str] = None
    original_claim: str = ""
    original_scope: str = "train_all"
    pattern_definition: str = ""
    accepted_despite_insufficient: bool = False


class ReviewState(BaseModel):

    data_fingerprint: str
    split_signature: Dict[str, int] = Field(default_factory=dict)
    gap_definition_digest: str = ""
    review_config: Dict[str, Any] = Field(default_factory=dict)
    sampling_seed: int = 42
    sampling_round: int = 1
    gaps: List[MissingPattern] = Field(default_factory=list)
    samples: List[SampleCard] = Field(default_factory=list)


class GapLoopRound(BaseModel):

    round_no: int
    digest_before: str = ""
    digest_after: str = ""
    verdicts: List[GapReview] = Field(default_factory=list)
    kept_gap_ids: List[str] = Field(default_factory=list)
    dropped_gap_ids: List[str] = Field(default_factory=list)
    added_gap_ids: List[str] = Field(default_factory=list)
    converged: bool = False
    degraded: bool = False
    sample_ids: List[int] = Field(default_factory=list)


class GapLoopTrace(BaseModel):

    max_rounds: int = 3
    rounds: List[GapLoopRound] = Field(default_factory=list)
    converged: bool = False
    stopped_reason: str = "budget"
    final_digest: str = ""
    final_gaps: List[MissingPattern] = Field(default_factory=list)


class GapLoopState(BaseModel):

    data_fingerprint: str = ""
    split_signature: Dict[str, int] = Field(default_factory=dict)
    final_gap_digest: str = ""
    review_config: Dict[str, Any] = Field(default_factory=dict)
    trace: GapLoopTrace = Field(default_factory=GapLoopTrace)
    final_gaps: List[MissingPattern] = Field(default_factory=list)


class FeatureSource(BaseModel):

    kind: Literal[
        "own_community_centroid",
        "donor_centroid",
        "blend_centroids",
        "mean_plus_noise",
        "text_gen",
        "permutation",
    ]
    donor_community: Optional[int] = Field(
        None, description="community whose centroid to borrow (donor_centroid / text_gen)"
    )
    target_community: Optional[int] = Field(
        None, description="community the node hides inside structurally"
    )
    second_community: Optional[int] = Field(
        None, description="second centroid community for blends / bridges"
    )
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="blend weight alpha*donor+(1-alpha)*target")
    noise_scale: float = Field(0.02, ge=0.0, le=0.5, description="Gaussian noise std on the feature")
    label: Optional[int] = Field(None, description="class label to assign; None = inherit target majority")


class EdgeTopology(BaseModel):

    kind: Literal[
        "attach_to_anchors",
        "bridge_two_communities",
        "star",
        "clique",
        "isolated",
        "single_edge",
    ]
    target_community: Optional[int] = Field(None, description="community for anchors / star hub")
    second_community: Optional[int] = Field(None, description="second endpoint community for bridges")
    edges_per_node: int = Field(2, ge=0, le=30, description="edges per new node for attach_to_anchors (le raised to 30 so hub-type specs can reach real anomaly degree levels, e.g. ~21 on History)")
    anchor_spread: bool = Field(True, description="True = distinct anchors per node; False = reuse one")
    extra_neighbor_prob: float = Field(
        0.0, ge=0.0, le=1.0, description="prob of adding a 2nd in-community neighbour (context-style)"
    )
    clique_size: Optional[int] = Field(None, description="nodes per clique (None = all new nodes)")


class AnchorSelection(BaseModel):

    pool: Literal["normal_nodes", "any_nodes"] = Field(
        "normal_nodes", description="restrict anchors to non-anomalous nodes when possible"
    )
    target_community: Optional[int] = Field(None, description="restrict pool to this community")
    second_community: Optional[int] = Field(None, description="second pool for bridges")
    spread_across_communities: bool = Field(
        True, description="round-robin anchors across communities (context-style diversity)"
    )


class AnomalySpec(BaseModel):

    family_name: str
    family_type: Literal["attribute", "structure", "both"] = Field(
        ..., description="drives attr_ano / stru_ano tagging at injection time"
    )
    feature_source: FeatureSource
    edge_topology: EdgeTopology = Field(default_factory=lambda: EdgeTopology(kind="attach_to_anchors"))
    anchor_selection: AnchorSelection = Field(default_factory=AnchorSelection)
    degree_cap: int = Field(3, ge=0, le=50, description="max edges per new node (realism; le raised to 50 to cover hub-type real anomaly degrees)")
    keep_degree_distribution: bool = True
    rationale: str = ""


class PatternPlan(BaseModel):

    pattern: str
    count: int = Field(..., ge=0)
    target_community: Optional[int] = Field(
        None, description="community id to attack, if applicable"
    )
    target_class: Optional[int] = Field(
        None, description="topic/class to mimic or hijack, if applicable"
    )
    transfer_attribute_from: Optional[int] = Field(
        None, description="class whose feature centroid to borrow for camouflage"
    )
    keep_degree_distribution: bool = True
    spec: Optional[AnomalySpec] = Field(
        None,
        description="Declarative construction spec for proposed/novel patterns; "
                    "executed by the SpecBuilder when no fixed-core builder exists.",
    )
    extra: Dict[str, Any] = Field(default_factory=dict)


class GlobalConstraints(BaseModel):
    preserve_homophily_of_normals: bool = True
    max_degree_delta: int = 2
    extra: Dict[str, Any] = Field(default_factory=dict)


class GenerationStrategy(BaseModel):

    total_new_anomalies: int = Field(..., ge=0)
    per_pattern: List[PatternPlan]
    global_constraints: GlobalConstraints = Field(default_factory=GlobalConstraints)
    rationale: str = ""


class NewNodeBatch(BaseModel):

    pattern: str
    num_nodes: int
    features: List[List[float]]
    new_edges: List[List[int]] = Field(
        default_factory=list,
        description="[[orig_node, new_local_idx], ...]; injector adds both directions",
    )
    labels: List[int] = Field(
        default_factory=list, description="topic/class label per new node"
    )
    anomaly_family: str = "unknown"
    family_type: Literal["attribute", "structure", "both", "unknown"] = "unknown"
    new_new_edges: List[List[int]] = Field(
        default_factory=list,
        description="[[local_i, local_j], ...] edges between two new nodes in this batch",
    )
    extra: Dict[str, Any] = Field(default_factory=dict)