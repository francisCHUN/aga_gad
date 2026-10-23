
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .review_sampling import SampleRecord, SamplingContext
from ..orchestration.schema import SampleCard, SampleFeedback, GapFeedback

log = logging.getLogger("aga.review.evidence")

NEIGHBOUR_TEXT_CHARS = 400
CARD_TEXT_CHARS = 4000
_NEIGHBOUR_SELECTION = "top-2 by degree + 1 seeded-random, self-loops excluded"

_VERSION_LIMITATION = (
    "text/feature version consistency unverified: csv text and mat features "
    "may come from different pipelines; text is context evidence only, never "
    "feature-level anomaly evidence"
)


_CHECKLISTS: Dict[str, List[str]] = {
    "context": [
        "FACT nbr_feat_cos of this card is well below normal_ctx_cos_mean "
        "(see family_facts) -- the feature deviation is established",
        "SEMANTIC: the neighbour content shows a concrete semantic conflict "
        "(quote the clashing fragments from node_text and neighbours)",
        "SEMANTIC: the deviation is substantial, beyond ordinary topical spread",
        "FACT/SEMANTIC split: no annotation already explains this node "
        "(label hidden; judge from evidence only)",
    ],
    "semantic_camouflage": [
        "FACT surface_similarity_nbr_cos vs normal_ctx_cos_mean and "
        "surface_matches_normal (family_facts) establish the surface side",
        "FACT feat_norm_train_percentile / degree_train_percentile give the "
        "independent-deviation side",
        "SEMANTIC: both evidence classes can hold simultaneously (citability "
        "check -- cite both fields)",
        "SEMANTIC: a real semantic clash exists on closer reading, not just style",
    ],
    "bridge": [
        "FACT n_communities_connected, edges_per_community and "
        "cross_community_ratio (family_facts) establish the connection split",
        "FACT own_community_control_median_cross gives the in-community contrast",
        "SEMANTIC: the cross-community links lack a reasonable explanation "
        "(quote what makes them unmotivated)",
        "SEMANTIC: not a legitimate cross-topic citation (judge from texts)",
    ],
    "hybrid": [
        "FACT the attribute side: deviation_direction and nbr_feat_cos vs "
        "labeled/normal bands (family_facts)",
        "FACT the structure side: degree_train_percentile and "
        "labeled_stru_degree_mean/p90",
        "SEMANTIC: both deviations hold simultaneously and are not covered by "
        "the labeled attribute/structure populations",
    ],
    "hub": [
        "FACT degree, degree_train_percentile, labeled_stru_degree_mean/p90 and "
        "degree_in_labeled_band establish the degree anomaly",
        "SEMANTIC: the hub positioning is anomalous, not just a popular core "
        "node of an on-topic cluster (judge from neighbour texts)",
        "SEMANTIC: does the gap definition still need its semantic condition, "
        "or is the degree fact the whole story?",
    ],
}
_DEFAULT_CHECKLIST = [
    "the node satisfies the pattern definition above",
    "the deviation is substantial beyond ordinary variation",
    "the evidence (metrics, neighbour context) supports the match",
    "when unsure, mark undecidable rather than guessing",
]


def _checklist_for(name: str) -> List[str]:
    low = (name or "").lower()
    for key, items in _CHECKLISTS.items():
        if key in low:
            return list(items)
    return list(_DEFAULT_CHECKLIST)


def _neighbour_block(ctx: SamplingContext, nid: int, rng) -> List[Tuple[int, str]]:
    nbr = ctx.neighbours(nid)
    if len(nbr) == 0:
        return []
    chosen: List[int] = []
    if len(nbr) >= 2:
        order = np.argsort(-ctx.degree[nbr], kind="stable")
        chosen.extend(int(nbr[i]) for i in order[:2])
    if len(chosen) < 3:
        rest = [int(x) for x in nbr if int(x) not in chosen]
        if rest:
            chosen.append(rest[int(rng.integers(0, len(rest)))])
    return chosen[:3]


def build_sample_cards(
    gaps: Sequence,
    records: Dict[str, List[SampleRecord]],
    ctx: SamplingContext,
    node_text: Dict[int, str],
) -> List[SampleCard]:

    cards: List[SampleCard] = []
    rng = np.random.default_rng(4242)
    for gid, recs in (records or {}).items():
        for rec in recs:
            nid = int(rec.sample_id)
            text = node_text.get(nid, "") or ""
            trunc = len(text) > CARD_TEXT_CHARS
            nbrs = _neighbour_block(ctx, nid, rng)
            nbr_texts = [(node_text.get(nb, "") or "")[:NEIGHBOUR_TEXT_CHARS]
                         for nb in nbrs]
            limitations = [_VERSION_LIMITATION]
            if not text:
                limitations.append("no text available for this node")
            if trunc:
                limitations.append("full text truncated for the html card; "
                                   "complete text is in full_cards.json")
            cards.append(SampleCard(
                sample_id=nid,
                gap_id=gid,
                group=rec.group,
                reason=rec.reason,
                stratum=rec.stratum,
                community=int(ctx.comm_id[nid]),
                degree=int(ctx.degree[nid]),
                cross_community_ratio=round(float(ctx.cross_ratio[nid]), 4),
                ctx_cos=(round(float(ctx.ctx_cos[nid]), 4)
                         if ctx.degree[nid] > 0 else None),
                degree_percentile=round(
                    ctx.train_percentile(ctx.degree, ctx.degree[nid]), 2),
                ctx_cos_percentile=(round(ctx.train_percentile(
                    ctx.ctx_cos, ctx.ctx_cos[nid]), 2)
                    if ctx.degree[nid] > 0 else None),
                cross_ratio_percentile=round(ctx.train_percentile(
                    ctx.cross_ratio, ctx.cross_ratio[nid]), 2),
                text=text[:CARD_TEXT_CHARS],
                text_truncated=trunc if text else None,
                neighbour_ids=[int(nb) for nb in nbrs],
                neighbour_texts=nbr_texts,
                neighbour_selection=_NEIGHBOUR_SELECTION,
                label=int(ctx.label_anomaly[nid]),
                label_source="train_labels",
                limitations=limitations,
            ))
    return cards


def occurrence_summary(
    cards: Sequence[SampleCard],
    feedback: Optional[Sequence[SampleFeedback]] = None,
) -> str:
    if not cards:
        return "no sample cards were drawn"
    if feedback is None:
        return (f"{len(cards)} sample cards awaiting human judgement "
                f"(pending human judgement; no occurrence claims before then)")
    counts = {"matches": 0, "not_matches": 0, "undecidable": 0}
    for f in feedback:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    tgt = sum(1 for c in cards if c.group == "targeted")
    return (f"judged: {counts['matches']} matches (existing instances: "
            f"counterexamples to 'completely missing'; NOT population ratios; "
            f"targeted cards are retrieval results, NOT population ratios), "
            f"{counts['not_matches']} not_matches (consistent with the claim, "
            f"weak support), {counts['undecidable']} undecidable/blank "
            f"(blank feedback never counts), from {len(cards)} cards "
            f"({tgt} targeted)")


def load_sample_feedback(path: str) -> List[SampleFeedback]:
    out: List[SampleFeedback] = []
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                sid = int(row.get("sample_id", ""))
            except ValueError:
                continue
            verdict = (row.get("verdict") or "").strip() or "undecidable"
            if verdict not in ("matches", "not_matches", "undecidable"):
                verdict = "undecidable"
            source = (row.get("source") or "").strip() or "human"
            if source not in ("human", "llm"):
                source = "human"
            out.append(SampleFeedback(
                gap_id=(row.get("gap_id") or "").strip(),
                sample_id=sid,
                verdict=verdict,
                comment=(row.get("comment") or "").strip(),
                source=source,
            ))
    return out


def load_gap_feedback(path: str) -> List[GapFeedback]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: List[GapFeedback] = []
    for entry in data.get("gaps", []):
        status = (entry.get("review_status") or "").strip() or "pending"
        if status not in ("pending", "confirmed", "revised_confirmed",
                          "rejected", "insufficient_evidence"):
            status = "pending"
        source = (entry.get("source") or "").strip() or "human"
        if source not in ("human", "llm"):
            source = "human"
        out.append(GapFeedback(
            gap_id=entry.get("gap_id", ""),
            review_status=status,
            revised_claim=entry.get("revised_claim"),
            reviewed_scope=entry.get("reviewed_scope"),
            human_comment=(entry.get("human_comment") or "").strip(),
            request_resample=bool(entry.get("request_resample", False)),
            source=source,
            coverage_criterion=entry.get("coverage_criterion"),
        ))
    return out


def _js_payload(data) -> str:
    blob = json.dumps(data, ensure_ascii=False)
    return (blob.replace("\\", "\\\\").replace('"', '\\"')
                .replace("\r", "\\r").replace("\n", "\\n")
                .replace("</", "<\\/"))


def export_review_package(
    review_dir: str,
    gaps: Sequence,
    cards: Sequence[SampleCard],
    evidence_summary: str,
    context_notes: Optional[Dict] = None,
    draft_key: str = "",
    prefill_sample: Optional[Sequence[SampleFeedback]] = None,
    prefill_gap: Optional[Sequence[GapFeedback]] = None,
) -> Dict[str, str]:
    os.makedirs(review_dir, exist_ok=True)
    context_notes = context_notes or {}

    csv_path = os.path.join(review_dir, "sample_feedback.csv")
    pre_ids = {(f.gap_id, f.sample_id): f for f in (prefill_sample or [])}
    has_prefill = bool(pre_ids)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["gap_id", "sample_id", "group", "verdict", "comment"]
        if has_prefill:
            header.append("source")
        w.writerow(header)
        for c in cards:
            row = [c.gap_id, c.sample_id, c.group, "", ""]
            pf = pre_ids.get((c.gap_id, c.sample_id))
            if has_prefill:
                if pf is not None:
                    row[3], row[4] = pf.verdict, pf.comment
                    row.append(pf.source)
                else:
                    row.append("human")

            w.writerow(row)

    gap_json_path = os.path.join(review_dir, "gap_feedback.json")
    pre_gap = {f.gap_id: f for f in (prefill_gap or [])}
    payload = {
        "instructions": (
            "Fill one entry per gap. review_status: confirmed | "
            "revised_confirmed (set revised_claim and reviewed_scope) | "
            "rejected | insufficient_evidence (set request_resample=true to "
            "draw one more round) | pending (no decision). Entries left "
            "pending NEVER pass to generation automatically."
        ),
        "gaps": [
            {
                "gap_id": g.gap_id,
                "pattern": g.name,
                "claim": g.claim,
                "review_status": (pre_gap[g.gap_id].review_status
                                  if g.gap_id in pre_gap else "pending"),
                "revised_claim": (pre_gap[g.gap_id].revised_claim
                                  if g.gap_id in pre_gap else None),
                "reviewed_scope": (pre_gap[g.gap_id].reviewed_scope
                                   if g.gap_id in pre_gap else None),
                "human_comment": (pre_gap[g.gap_id].human_comment
                                  if g.gap_id in pre_gap else ""),
                "request_resample": (pre_gap[g.gap_id].request_resample
                                     if g.gap_id in pre_gap else False),
                "source": (pre_gap[g.gap_id].source
                           if g.gap_id in pre_gap else "human"),
                "coverage_criterion": (pre_gap[g.gap_id].coverage_criterion
                                       if g.gap_id in pre_gap else None),
            }
            for g in gaps
        ],
    }
    with open(gap_json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    full_path = os.path.join(review_dir, "full_cards.json")
    with open(full_path, "w", encoding="utf-8") as fh:
        json.dump([c.model_dump() for c in cards], fh, ensure_ascii=False, indent=2)

    html_path = os.path.join(review_dir, "review_report.html")
    data = {
        "draftKey": draft_key,
        "evidenceSummary": evidence_summary,
        "contextNotes": context_notes,
        "gaps": [
            {
                "gap_id": g.gap_id,
                "pattern": g.name,
                "claim": g.claim,
                "scope": g.scope,
                "pattern_definition": g.pattern_definition,
                "evidence_needed": list(g.evidence_needed),
                "checklist": _checklist_for(g.name),
            }
            for g in gaps
        ],
        "cards": [c.model_dump() for c in cards],
        "llmSample": {f"{f.gap_id}:{f.sample_id}": f.verdict
                      for f in (prefill_sample or [])},
        "llmGap": {f.gap_id: f.review_status for f in (prefill_gap or [])},
    }
    html = _render_html(data)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return {"html": html_path, "sample_csv": csv_path, "gap_json": gap_json_path,
            "full_cards": full_path}


def _render_html(data: Dict) -> str:
    payload = _js_payload(data)
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AGA gap review workbench</title>
<style>
body{font-family:-apple-system,sans-serif;margin:2rem auto;max-width:72rem;color:
h1{font-size:1.4rem;} h2{font-size:1.1rem;margin:.2rem 0;} h3{font-size:.95rem;}
.gap{border:1px solid
.meta{font-size:.9rem;margin:.25rem 0;} .meta b{color:
.card{border-left:4px solid
.card.targeted{border-color:
.card.boundary{border-color:
.group{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:
.numerics{font-family:ui-monospace,monospace;font-size:.83rem;margin:.2rem 0;}
.badge{display:inline-block;padding:0 .4rem;border-radius:6px;font-size:.75rem;margin:0 .15rem;background:
.badge.ext-low{background:
pre{white-space:pre-wrap;font-size:.85rem;margin:.25rem 0;}
details summary{cursor:pointer;color:
.hint{color:
.controls{margin:.4rem 0;} .controls label{margin-right:1rem;}
.footer{margin-top:2rem;border-top:1px solid
button{font:inherit;padding:.3rem .8rem;border-radius:6px;border:1px solid
.radiorow label{display:inline-block;margin-right:.9rem;}
</style>
</head>
<body>
<h1>AGA gap review workbench</h1>
<div class="meta">Gap Analysis output is a set of <b>hypotheses</b>. Judge the sampled
real TRAINING nodes against each pattern's definition, then export the feedback
files and drop them next to this html. Progress: <span id="judgedcount">0</span>/<span id="totalcount">0</span> judged.</div>
<div id="progressouter"><div id="progressfill"></div></div>
<h3>What counts as matching</h3>
<div id="checklistpanel"></div>
<h3>How to use this page</h3>
<div class="hint">For each card pick a verdict (matches = the node IS an instance of
the pattern = a counterexample to 'missing'; not_matches = consistent with the
claim; undecidable = cannot tell). Optionally comment. At the end, choose each
gap's conclusion and click <b>Export feedback files</b>: this downloads
<b>sample_feedback.csv</b> and <b>gap_feedback.json</b> ready for
<code>--resume-review</code>. Drafts autosave to your browser (Reset draft to
clear). Training labels are hidden per card -- open 'reveal train labels' only
after judging.</div>
<div id="contextpanel"></div>
<div id="gaps"></div>
<div class="footer">
<button id="exportbtn">Export feedback files</button>
<button id="resetbtn">Reset draft</button>
<span class="hint" id="exportnote"></span>
</div>
<script>
const DATA = JSON.parse("__PAYLOAD__");
const esc = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;');
const radio = s => s;
// gap conclusion radio states, enumerated as plain "..." literals:
// "pending", "confirmed", "revised_confirmed", "rejected", "insufficient_evidence"
const GAP_STATUSES = ["pending", "confirmed", "revised_confirmed", "rejected", "insufficient_evidence"];
const badge = (label, pct) => {
  if (pct === null || pct === undefined) return `<span class="badge">${label} n/a</span>`;
  let cls = '';
  if (pct < 5) cls = 'ext-low'; else if (pct >= 95) cls = 'ext-high';
  return `<span class="badge ${cls}">${label} ${pct.toFixed(1)}% of training nodes</span>`;
};
// context panel (reference profile + LLM pre-review summary)
(() => {
  const cn = DATA.contextNotes || {};
  const host = document.getElementById('contextpanel');
  const ref = cn.labeled_anomaly_reference || cn.reference_profile || null;
  let html = '';
  if (ref) {
    html += `<h3>Labeled-anomaly reference profile (training labels)</h3><pre>${esc(JSON.stringify(ref, null, 1))}</pre>`;
  }
  if (cn.unmeasured_note) html += `<div class="hint">${esc(cn.unmeasured_note)}</div>`;
  if (cn.llm_precheck) html += `<div class="hint"><b>LLM suggestion summary:</b> ${esc(cn.llm_precheck)}</div>`;
  host.innerHTML = html;
})();
// checklist panel: per-family "what counts as matching"
(() => {
  const host = document.getElementById('checklistpanel');
  const items = [];
  for (const g of DATA.gaps) {
    items.push(`<div class="meta"><b>${esc(g.gap_id)} ${esc(g.pattern)}</b>:
<ol>${(g.checklist||[]).map(c=>`<li>${esc(c)}</li>`).join('')}</ol></div>`);
  }
  host.innerHTML = items.join('');
})();
const state = { drafts: {}, gapChoice: {} };
const LSKEY = 'aga-review-draft-' + (DATA.draftKey || 'nodefault');
try { Object.assign(state.drafts, JSON.parse(localStorage.getItem(LSKEY) || '{}')); } catch(e) {}
const saveDraft = () => { try { localStorage.setItem(LSKEY, JSON.stringify(state.drafts)); } catch(e) {} };
document.getElementById('resetbtn').onclick = () => {
  localStorage.removeItem(LSKEY); state.drafts = {}; state.gapChoice = {};
  location.reload();
};
// cards per gap
const gapsHost = document.getElementById('gaps');
let total = 0, judged = 0;
const updateProgress = () => {
  document.getElementById('judgedcount').textContent = judged;
  document.getElementById('totalcount').textContent = total;
  document.getElementById('progressfill').style.width = (total ? 100*judged/total : 0) + '%';
};
for (const g of DATA.gaps) {
  const sec = document.createElement('section'); sec.className = 'gap';
  const cards = DATA.cards.filter(c => c.gap_id === g.gap_id);
  const llmGapSug = (DATA.llmGap || {})[g.gap_id];
  let inner = `<h2>${esc(g.gap_id)} — ${esc(g.pattern)}</h2>
  <div class="meta"><b>claim:</b> ${esc(g.claim)} <b>scope:</b> ${esc(g.scope)}</div>
  <div class="meta"><b>definition:</b> ${esc(g.pattern_definition)}</div>
  <div class="meta"><b>LLM suggestion:</b> ${llmGapSug ? esc(llmGapSug) : 'none'}</div>`;
  inner += `<div class="radiorow"><b>conclusion:</b>` +
    ['confirmed','revised_confirmed','rejected','insufficient_evidence','pending']
      .map(s => `<label><input type="radio" name="status-${esc(g.gap_id)}" value="${s}"
      ${state.gapChoice[g.gap_id]===s?'checked':''}>${radio(s)}</label>`).join('') + `</div>`;
  inner += `<div class="controls">
    <label>revised claim: <input id="revclaim-${esc(g.gap_id)}" size="40"></label>
    <label>reviewed scope: <input id="revscope-${esc(g.gap_id)}" size="12"></label>
    <label>request resample: <input type="checkbox" id="resample-${esc(g.gap_id)}"></label>
    <label>coverage criterion: <input id="crit-${esc(g.gap_id)}" size="30"></label>
  </div>
  <div><textarea id="comment-${esc(g.gap_id)}" rows="2" cols="80"
    placeholder="gap-level comment (human_comment)"></textarea></div>`;
  for (const c of cards) {
    total++;
    const key = g.gap_id + ':' + c.sample_id;
    const llmSug = (DATA.llmSample || {})[key];
    const draft = state.drafts[key] || '';
    const verdictVal = draft || llmSug || '';
    if (verdictVal && verdictVal !== 'undecidable') judged++;
    const nbs = (c.neighbour_ids||[]).map((id,i) =>
      `<div class="nb">neighbour
       <pre>${esc((c.neighbour_texts||[])[i]||'(no text)')}</pre></div>`).join('');
    inner += `<div class="card ${c.group}" id="card-${esc(key)}">
      <span class="group">${esc(c.group)} · ${esc(c.reason)}</span>
      <div class="numerics">node
        ${badge('degree', c.degree_percentile)} ${badge('ctx_cos', c.ctx_cos_percentile)}
        ${badge('cross_ratio', c.cross_ratio_percentile)}
        ${llmSug ? `<span class="badge">LLM suggestion: ${esc(llmSug)}</span>` : ''}</div>
      <details><summary>node text ${c.text_truncated?'(truncated; full text in full_cards.json)':''}</summary>
        <pre>${esc(c.text || '(no text)')}</pre></details>
      <details><summary>neighbours (${esc(c.neighbour_selection)})</summary>${nbs}</details>
      <details><summary>reveal train labels</summary>
        anomaly=${c.label} (source: ${esc(c.label_source)})</details>
      <div class="hint">limitations: ${(c.limitations||[]).map(esc).join('; ')}</div>
      <div class="controls">
        verdict: <select data-verdict="${esc(key)}" class="verdictsel">
          ${['','matches','not_matches','undecidable'].map(v =>
            `<option value="${v}" ${verdictVal===v?'selected':''}>${v||'(undecided)'}</option>`).join('')}
        </select>
        comment: <input data-comment="${esc(key)}" size="40" value="">
      </div>
    </div>`;
  }
  sec.innerHTML = inner;
  // wire inputs
  sec.querySelectorAll('select[data-verdict]').forEach(sel => {
    sel.addEventListener('change', () => {
      const k = sel.getAttribute('data-verdict');
      const was = state.drafts[k] && state.drafts[k] !== 'undecidable';
      const now = sel.value && sel.value !== 'undecidable';
      if (was && !now) judged--; if (!was && now) judged++;
      state.drafts[k] = sel.value; saveDraft(); updateProgress();
    });
  });
  sec.querySelectorAll('input[data-comment]').forEach(inp => {
    inp.addEventListener('input', () => {
      state.drafts['c:' + inp.getAttribute('data-comment')] = inp.value; saveDraft();
    });
  });
  sec.querySelectorAll('input[type=radio]').forEach(rd => {
    rd.addEventListener('change', () => { state.gapChoice[g.gap_id] = rd.value; saveDraft(); });
  });
  gapsHost.appendChild(sec);
}
updateProgress();
// export: build csv + json exactly like the legacy templates
document.getElementById('exportbtn').onclick = () => {
  const rows = [['gap_id','sample_id','group','verdict','comment','source']];
  for (const c of DATA.cards) {
    const k = c.gap_id + ':' + c.sample_id;
    const v = state.drafts[k] || 'undecidable';
    const llm = (DATA.llmSample || {})[k];
    const src = (state.drafts[k] && state.drafts[k] !== (llm||'undecidable')) ? 'human'
                : (llm ? 'llm' : 'human');
    const cm = state.drafts['c:' + k] || '';
    rows.push([c.gap_id, c.sample_id, c.group, v, cm, src]);
  }
  const csv = rows.map(r => r.map(x => `"${String(x).replace(/"/g,'""')}"`).join(',')).join('\n');
  const gapsOut = { instructions: 'exported from the review workbench', gaps: [] };
  for (const g of DATA.gaps) {
    const gc = state.gapChoice[g.gap_id] || ((DATA.llmGap||{})[g.gap_id] || 'pending');
    const src = (state.gapChoice[g.gap_id] && state.gapChoice[g.gap_id] !== (DATA.llmGap||{})[g.gap_id]) ? 'human' : 'llm';
    gapsOut.gaps.push({
      gap_id: g.gap_id, pattern: g.pattern, claim: g.claim,
      review_status: gc,
      revised_claim: (document.getElementById('revclaim-'+g.gap_id)||{}).value || null,
      reviewed_scope: (document.getElementById('revscope-'+g.gap_id)||{}).value || null,
      human_comment: (document.getElementById('comment-'+g.gap_id)||{}).value || '',
      request_resample: !!((document.getElementById('resample-'+g.gap_id)||{}).checked),
      source: gc === 'pending' && !(DATA.llmGap||{})[g.gap_id] ? 'human' : src,
      coverage_criterion: (document.getElementById('crit-'+g.gap_id)||{}).value || null,
    });
  }
  const download = (name, text, type) => {
    const blob = new Blob([text], {type});
    const url = URL.createObjectURL  ? URL.createObjectURL(blob) : window.webkitURL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };
  download('sample_feedback.csv', csv, 'text/csv');
  download('gap_feedback.json', JSON.stringify(gapsOut, null, 2), 'application/json');
  document.getElementById('exportnote').textContent =
    ' exported sample_feedback.csv + gap_feedback.json (see your downloads)';
};
</script>
</body>
</html>"""
    html = html.replace("__PAYLOAD__", payload)
    return html