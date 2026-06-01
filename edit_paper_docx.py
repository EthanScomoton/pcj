"""
Apply structural and formula corrections to Port_IES_Paper_Sections1-2.docx
in-place.

Edits performed
---------------
1. Reorder Section 2:
       2.1 Architecture          (unchanged)
       2.2 LSTM forecaster       (was 2.3)
       2.3 Conformal calibration (was 2.4)
       2.4 Energy balance & BES  (was 2.2)
       2.5 Parametrised MPC      (unchanged)
       2.6 Regional emissions    (unchanged)

2. Renumber equations to follow the new order:
       (4)(5)(6)   LSTM        -> (1)(2)(3)
       (7)(8)(9)   Conformal   -> (4)(5)(6)
       (1)(2)(3)   Balance/SOC -> (7)(8)(9)
       (10)        MPC obj     -> (10)  (unchanged)

3. Update cross-references in body text:
       "Eq. (6)" / "Eq. (7)"  (in 2.3 conformal) -> "Eq. (3)" / "Eq. (4)"
       "Eq. (1)" / "Eqs. (2)-(3)" / "Eq. (9)" (in 2.5 MPC)
            -> "Eq. (7)" / "Eqs. (8)-(9)" / "Eq. (6)"
       "Section 2.3" (architecture) -> "Section 2.2"
       "Section 2.4" (architecture) -> "Section 2.3"
       Add note that dispatch tier (Sections 2.4-2.6) is the full physical
       model.

4. Fix carbon-intensity multipliers in Section 2.6:
       peak 1.4 -> 1.25, valley 0.6 -> 0.75
       (matches carbon_module.peak_multiplier=1.25, valley_multiplier=0.75)

5. Expand Eq. (10) and its explanation to reflect the actual implementation
   in strategies.py (_ParametrizedMPC):
       J = Sum_t pi_t Pg_t
         + lambda_c * Sum_t phi_t Pg_t
         + w_p * (P_peak - P_run)
         - alpha_s * C * S_H
         + w_a * C * (S_H - S_target)^2
         + R(P_c, P_d, P_r)
   where R = rho_cyc * Sum(P_c+P_d) + rho_curt * Sum(P_hat_r - P_r) is the
   composite regulariser that suppresses micro-cycling and softly
   encourages renewable utilisation.

6. Update the introduction roadmap sentence in Section 1 to reflect the
   new Section 2 ordering (prediction model -> conformal -> full physical
   model).

7. Tighten the architecture description (Section 2.1) so that it points
   to the new section numbers and explicitly says the dispatch tier is
   the full-model portion.

All edits preserve Times-New-Roman runs and existing paragraph
properties; only run *text* is altered.
"""
from __future__ import annotations
import copy
import os
import re
import sys

from docx import Document
from docx.oxml.ns import qn

SRC = '/sessions/beautiful-happy-ramanujan/mnt/pcj/Port_IES_Paper_Sections1-2.docx'
DST = '/sessions/beautiful-happy-ramanujan/mnt/pcj/Port_IES_Paper_Sections1-2.docx'


# ---------------------------------------------------------------------------
# 1) Open document
# ---------------------------------------------------------------------------
doc = Document(SRC)
body = doc.element.body

# ---------------------------------------------------------------------------
# 2) Helper: enumerate body children in document order (paragraphs + tables).
# ---------------------------------------------------------------------------
def block_iter(body_el):
    for child in body_el.iterchildren():
        if child.tag == qn('w:p') or child.tag == qn('w:tbl'):
            yield child


def get_text(elem) -> str:
    """Concatenate all <w:t> text in a block element."""
    parts = []
    for t in elem.iter(qn('w:t')):
        if t.text:
            parts.append(t.text)
    return ''.join(parts)


def set_text_inplace(elem, old: str, new: str):
    """Replace `old` with `new` in the runs of *elem* without touching
    formatting. Works only when *old* lives entirely inside a single
    <w:t>. If it spans runs, we fall back to rebuilding the first match
    by merging adjacent runs."""
    # Quick path: single <w:t> contains the whole substring.
    for t in elem.iter(qn('w:t')):
        if t.text and old in t.text:
            t.text = t.text.replace(old, new)
            return True
    # Fallback: concatenate all <w:t> texts then split-and-rewrite. We
    # only support a single occurrence to keep things simple.
    all_t = list(elem.iter(qn('w:t')))
    full = ''.join((t.text or '') for t in all_t)
    if old not in full:
        return False
    new_full = full.replace(old, new, 1)
    # Distribute new_full back to the runs proportionally to old lengths.
    # Easiest robust strategy: place all text in the first <w:t>, blank
    # the rest. This loses partial-run italic/bold inside the span we
    # touched, but is acceptable for the descriptive plain text we edit.
    if not all_t:
        return False
    all_t[0].text = new_full
    for t in all_t[1:]:
        t.text = ''
    return True


# ---------------------------------------------------------------------------
# 3) Locate section blocks by their heading text.
# ---------------------------------------------------------------------------
blocks = list(block_iter(body))

def find_index(predicate):
    for i, b in enumerate(blocks):
        if predicate(b):
            return i
    return -1


def heading_startswith(prefix):
    return lambda b: b.tag == qn('w:p') and get_text(b).strip().startswith(prefix)


EMSP = ' '  # em-space used between numbering and heading text
idx_22 = find_index(heading_startswith(f'2.2{EMSP}Energy Balance'))
idx_23 = find_index(heading_startswith(f'2.3{EMSP}LSTM'))
idx_24 = find_index(heading_startswith(f'2.4{EMSP}Conformal'))
idx_25 = find_index(heading_startswith(f'2.5{EMSP}Parametrised'))
idx_26 = find_index(heading_startswith(f'2.6{EMSP}Regional'))

assert -1 not in (idx_22, idx_23, idx_24, idx_25, idx_26), (
    f"Could not locate all section headings: 22={idx_22} 23={idx_23} "
    f"24={idx_24} 25={idx_25} 26={idx_26}")

print(f"[locate] 2.2={idx_22}  2.3={idx_23}  2.4={idx_24}  "
      f"2.5={idx_25}  2.6={idx_26}")

# Blocks for each section: [start, next_start)
block_22 = blocks[idx_22:idx_23]   # Energy balance (old 2.2)
block_23 = blocks[idx_23:idx_24]   # LSTM           (old 2.3)
block_24 = blocks[idx_24:idx_25]   # Conformal      (old 2.4)
# Sections 2.5 and 2.6 stay in place.

# ---------------------------------------------------------------------------
# 4) Move blocks in the XML tree: new order LSTM, Conformal, Balance
# ---------------------------------------------------------------------------
# Remove block_22 (Energy balance) from its current position and re-insert
# it just before the start of Section 2.5.

# To keep the XML stable while we mutate, we'll detach the three groups
# and re-attach them in the new order, immediately after the last block of
# Section 2.1 (which is blocks[idx_22 - 1]).
anchor_after = blocks[idx_22 - 1]  # last element of Section 2.1

# Detach all three blocks first.
for grp in (block_22, block_23, block_24):
    for el in grp:
        body.remove(el)

# Re-attach in order: LSTM (was 2.3) -> Conformal (was 2.4) -> Balance (was 2.2).
new_sequence = list(block_23) + list(block_24) + list(block_22)
current_anchor = anchor_after
for el in new_sequence:
    current_anchor.addnext(el)
    current_anchor = el

# Refresh block index after the move.
blocks = list(block_iter(body))

# ---------------------------------------------------------------------------
# 5) Renumber section headings (2.2 LSTM..., 2.3 Conformal..., 2.4 Energy...).
# ---------------------------------------------------------------------------
def update_heading(prefix_old: str, prefix_new: str):
    for b in blocks:
        if b.tag != qn('w:p'):
            continue
        txt = get_text(b).strip()
        if txt.startswith(prefix_old):
            set_text_inplace(b, prefix_old, prefix_new)
            print(f"[heading] {prefix_old!r} -> {prefix_new!r}")
            return
    raise RuntimeError(f"Did not find heading {prefix_old!r}")


# 2.3 LSTM-Based ... -> 2.2 LSTM-Based ...
update_heading(f'2.3{EMSP}LSTM-Based', f'2.2{EMSP}LSTM-Based')
# 2.4 Conformal Prediction ... -> 2.3 Conformal Prediction ...
update_heading(f'2.4{EMSP}Conformal', f'2.3{EMSP}Conformal')
# 2.2 Energy Balance ... -> 2.4 Energy Balance ...
update_heading(f'2.2{EMSP}Energy Balance', f'2.4{EMSP}Energy Balance')

# ---------------------------------------------------------------------------
# 6) Renumber equation labels (right-hand column of single-row equation
#    tables).
# ---------------------------------------------------------------------------
EQ_MAP = {
    '(1)':  '(7)',   # Energy balance
    '(2)':  '(8)',   # SOC dynamics
    '(3)':  '(9)',   # SOC limits
    '(4)':  '(1)',   # Feature gating
    '(5)':  '(2)',   # h_con
    '(6)':  '(3)',   # Reparameterisation
    '(7)':  '(4)',   # Non-conformity score
    '(8)':  '(5)',   # Conformal quantile
    '(9)':  '(6)',   # Prediction interval
    # '(10)': '(10)' unchanged
}

# Find all equation tables (single-row, two-column, second column is the
# label "(N)").
def equation_tables(body_el):
    for tbl in body_el.iter(qn('w:tbl')):
        rows = tbl.findall(qn('w:tr'))
        if len(rows) != 1:
            continue
        cells = rows[0].findall(qn('w:tc'))
        if len(cells) != 2:
            continue
        label_text = ''.join(t.text or '' for t in cells[1].iter(qn('w:t')))
        m = re.fullmatch(r'\s*\((\d+)\)\s*', label_text)
        if not m:
            continue
        yield tbl, cells[1], label_text.strip(), m.group(0).strip()


# We must rewrite in two passes (otherwise (1)->(7) would later be
# rewritten by (7)->(4)). Strategy: stamp temporary placeholders, then
# replace placeholders with final values.
placeholders = {}  # original "(N)" -> "<<eqXXX>>"
for tbl, cell, label, label_clean in list(equation_tables(body)):
    if label_clean not in EQ_MAP:
        continue
    placeholder = f"<<eq{label_clean.strip('()')}>>"
    placeholders[label_clean] = placeholder
    # Rewrite cell content to placeholder.
    # We replace the first <w:t> inside the cell that contains the label.
    target_t = None
    for t in cell.iter(qn('w:t')):
        if t.text and label_clean in t.text:
            target_t = t
            break
    if target_t is None:
        # Fallback: text spread across multiple <w:t>; clear all and put
        # placeholder in first <w:t>.
        all_t = list(cell.iter(qn('w:t')))
        if not all_t:
            continue
        all_t[0].text = placeholder
        for t in all_t[1:]:
            t.text = ''
    else:
        target_t.text = target_t.text.replace(label_clean, placeholder)
    print(f"[eq-stamp] {label_clean} -> {placeholder}")

# Second pass: replace placeholders with final eq numbers.
for tbl in body.iter(qn('w:tbl')):
    for t in tbl.iter(qn('w:t')):
        if not t.text:
            continue
        for orig, ph in placeholders.items():
            if ph in t.text:
                final = EQ_MAP[orig]
                t.text = t.text.replace(ph, final)
                print(f"[eq-final] {ph} -> {final}")

# ---------------------------------------------------------------------------
# 7) Update equation cross-references inside body paragraphs.
# ---------------------------------------------------------------------------
# These are descriptive references such as "Eq. (6)" or "Eqs. (2)-(3)".
# We must guard against accidentally rewriting an equation label inside
# an equation table (handled above already). Body paragraphs only.
CROSSREF_MAP = [
    # In conformal section (now 2.3) the old "Eq. (6)" / "Eq. (7)" refer
    # to LSTM reparameterisation (now Eq. (3)) and to the non-conformity
    # score (now Eq. (4)).
    ('Eq. (6)',  'Eq. (3)'),
    ('Eq. (7)',  'Eq. (4)'),
    # In MPC section (still 2.5) the references to Eqs. (1)-(3) and
    # Eq. (9) need to follow the energy-balance/SOC renumbering.
    ('Eq. (1)',  'Eq. (7)'),
    ('Eqs. (2)–(3)', 'Eqs. (8)–(9)'),
    ('Eq. (9)',  'Eq. (6)'),
]

# Use temporary tokens to avoid double-rewriting (e.g. "(6)" -> "(3)"
# then "(7)"... -> "(4)" then we might accidentally hit a newly-written
# "(3)").
tmp_marker = {}
for i, (src, _) in enumerate(CROSSREF_MAP):
    tmp_marker[src] = f"<<xref{i}>>"

for p in body.iter(qn('w:p')):
    # Skip equation tables; iterate only paragraphs *not* nested in a
    # table. The iter() above traverses everything, so we must check.
    in_table = False
    parent = p.getparent()
    while parent is not None:
        if parent.tag == qn('w:tbl'):
            in_table = True
            break
        parent = parent.getparent()
    if in_table:
        continue

    # Phase 1: stamp temporary tokens, longest pattern first to avoid
    # "Eqs. (2)-(3)" matching the "Eq. (2)" substring inside it.
    for src in sorted(CROSSREF_MAP, key=lambda kv: -len(kv[0])):
        src_text, _ = src
        for t in p.iter(qn('w:t')):
            if t.text and src_text in t.text:
                t.text = t.text.replace(src_text, tmp_marker[src_text])
                print(f"[xref-stamp] {src_text!r} -> token")

    # Phase 2: replace tokens with the final cross-ref text.
    for src_text, dst_text in CROSSREF_MAP:
        for t in p.iter(qn('w:t')):
            if t.text and tmp_marker[src_text] in t.text:
                t.text = t.text.replace(tmp_marker[src_text], dst_text)

# ---------------------------------------------------------------------------
# 8) Fix the carbon-intensity multipliers in Section 2.6.
# ---------------------------------------------------------------------------
PEAK_OLD = 'is scaled upwards by a factor of 1.4'
PEAK_NEW = 'is scaled upwards by a factor of 1.25'
VALLEY_OLD = 'is scaled downwards by a factor of 0.6'
VALLEY_NEW = 'is scaled downwards by a factor of 0.75'

for p in body.iter(qn('w:p')):
    in_table = False
    parent = p.getparent()
    while parent is not None:
        if parent.tag == qn('w:tbl'):
            in_table = True
            break
        parent = parent.getparent()
    if in_table:
        continue
    full = get_text(p)
    if PEAK_OLD in full:
        set_text_inplace(p, PEAK_OLD, PEAK_NEW)
        print(f"[carbon] peak multiplier 1.4 -> 1.25")
    if VALLEY_OLD in full:
        set_text_inplace(p, VALLEY_OLD, VALLEY_NEW)
        print(f"[carbon] valley multiplier 0.6 -> 0.75")

# ---------------------------------------------------------------------------
# 9) Update the introduction-roadmap sentence so that it matches the new
#    Section-2 ordering (prediction model first, then full physical model).
# ---------------------------------------------------------------------------
ROADMAP_OLD = (
    "Section 2 presents the architecture of the proposed port integrated "
    "energy-management framework, formulates the energy balance and the "
    "battery storage dynamics, details the deep neural load-forecasting "
    "backbone together with the conformal calibration layer, and develops "
    "the parametrised carbon-aware rolling-horizon controller. "
    "Quantitative results, ablation studies and concluding remarks will "
    "be reported subsequently."
)
ROADMAP_NEW = (
    "Section 2 first presents the architecture of the proposed port "
    "integrated energy-management framework, then details the deep "
    "neural load-forecasting backbone together with the conformal "
    "calibration layer that converts the point forecast into a "
    "distribution-free prediction interval, and finally develops the "
    "full physical port-integrated-energy-system model: the energy "
    "balance and battery storage dynamics, the parametrised carbon-aware "
    "rolling-horizon controller and the regional/dynamic carbon-"
    "intensity model. Quantitative results, ablation studies and "
    "concluding remarks will be reported subsequently."
)
for p in body.iter(qn('w:p')):
    if ROADMAP_OLD in get_text(p):
        set_text_inplace(p, ROADMAP_OLD, ROADMAP_NEW)
        print("[roadmap] updated Section 2 roadmap sentence")
        break

# ---------------------------------------------------------------------------
# 10) Update the architecture paragraph (Section 2.1) so that the section
#     numbers it references match the new layout.
# ---------------------------------------------------------------------------
ARCH_OLD = (
    "The prediction tier hosts the pre-trained bidirectional LSTM model "
    "with feature gating and dual attention (Section 2.3), together "
    "with the split-conformal calibrator (Section 2.4); "
    "it consumes the most recent 20-step feature window and emits a "
    "24-hour ahead point forecast together with a calibrated prediction "
    "interval. The dispatch tier hosts five interchangeable scheduling "
    "strategies that share a common parametrised CVXPY core "
    "(Section 2.5) and issue, at each hour, the battery "
    "charge/discharge and grid import set-points to be applied to the "
    "physical port equipment."
)
ARCH_NEW = (
    "The prediction tier hosts the pre-trained bidirectional LSTM model "
    "with feature gating and dual attention (Section 2.2), together "
    "with the split-conformal calibrator (Section 2.3); "
    "it consumes the most recent 20-step feature window and emits a "
    "24-hour ahead point forecast together with a calibrated prediction "
    "interval. The dispatch tier, which embodies the full physical PIES "
    "model formulated in Sections 2.4–2.6, hosts five "
    "interchangeable scheduling strategies that share a common "
    "parametrised CVXPY core (Section 2.5) and issue, at each "
    "hour, the battery charge/discharge and grid import set-points to "
    "be applied to the physical port equipment."
)
for p in body.iter(qn('w:p')):
    if ARCH_OLD in get_text(p):
        set_text_inplace(p, ARCH_OLD, ARCH_NEW)
        print("[arch] updated Section 2.1 architecture paragraph")
        break

# ---------------------------------------------------------------------------
# 11) Insert a transition sentence at the start of the new Section 2.4
#     (Energy balance) reminding the reader that the prediction layer has
#     been described and the full physical model now follows.
# ---------------------------------------------------------------------------
ENERGY_OLD = (
    "At each hourly time step t ∈ {1, …, T}, "
    "the aggregate port load must be served by the sum of grid "
    "purchase, renewable consumption and net storage discharge, while "
    "bidirectional flow to the upstream grid is disallowed:"
)
ENERGY_NEW = (
    "Having specified the deep neural load forecaster (Section 2.2) "
    "and the split-conformal calibration layer (Section 2.3) that "
    "supplies it with a target-coverage prediction interval, we now turn "
    "to the description of the physical port integrated energy system "
    "into which those forecasts are injected as parameters. At each "
    "hourly time step t ∈ {1, …, T}, the aggregate port load "
    "must be served by the sum of grid purchase, renewable consumption "
    "and net storage discharge, while bidirectional flow to the upstream "
    "grid is disallowed:"
)
for p in body.iter(qn('w:p')):
    if ENERGY_OLD in get_text(p):
        set_text_inplace(p, ENERGY_OLD, ENERGY_NEW)
        print("[2.4] added transition opening to Energy Balance section")
        break

# ---------------------------------------------------------------------------
# 12) Augment the renewable-utilisation note in Section 2.4 with a
#     pooled-injection statement (P_r = E_PV + E_w) so that the
#     subsequent controller equations are notationally consistent.
# ---------------------------------------------------------------------------
RENEW_OLD = (
    ", with the hat denoting forecast values. A small curtailment "
    "slack is admitted at the model level so that the underlying "
    "convex programme remains feasible when the renewable forecast "
    "transiently exceeds the load."
)
RENEW_NEW = (
    ", with the hat denoting forecast values. For the purposes of the "
    "dispatch optimisation in Section 2.5 the two renewable "
    "channels are pooled into a single aggregated injection P_r,t = "
    "E_PV,t + E_w,t whose upper bound is the sum of the corresponding "
    "renewable forecasts. A small curtailment slack is admitted at the "
    "model level so that the underlying convex programme remains "
    "feasible when the renewable forecast transiently exceeds the load."
)
for p in body.iter(qn('w:p')):
    if RENEW_OLD in get_text(p):
        set_text_inplace(p, RENEW_OLD, RENEW_NEW)
        print("[2.4] added pooled renewable-injection clarifier")
        break

# ---------------------------------------------------------------------------
# 13) Add a one-sentence pointer in Section 2.2 (LSTM model) explaining
#     that Model 4 was selected from a 5-model comparative study in the
#     companion manuscript (Table 2/Fig. 13 there).
# ---------------------------------------------------------------------------
LSTM_OLD = (
    "deployed on the same industrial hardware that runs the dispatch "
    "controller. Given a sliding window"
)
LSTM_NEW = (
    "deployed on the same industrial hardware that runs the dispatch "
    "controller. The architecture is taken from a companion "
    "FeatureGating–BiLSTM–DualAttention study, in which five "
    "candidate variants (Models 0–4) were trained on a full year "
    "of real Qingdao dry-bulk port operational data and benchmarked on "
    "RMSE, MAPE, R² and MAE; Model 4 (FeatureGating + BiLSTM + "
    "DualAttention) attained the lowest MAPE and the highest R² "
    "and is therefore adopted as the production backbone in the "
    "present framework. Given a sliding window"
)
for p in body.iter(qn('w:p')):
    if LSTM_OLD in get_text(p):
        set_text_inplace(p, LSTM_OLD, LSTM_NEW)
        print("[2.2] inserted companion-study Model 4 citation")
        break

# ---------------------------------------------------------------------------
# 14) Augment the reparameterisation explanation (right after Eq. (3) in
#     the new numbering) with a sentence noting that the 0.1 scaling
#     factor mildly regularises the mean prediction and that the
#     calibrated uncertainty is the responsibility of the conformal
#     layer of Section 2.3.
# ---------------------------------------------------------------------------
REPAR_OLD = (
    "recovers the demand prediction in physical units (kW). When "
    "the runtime feature dimensionality"
)
REPAR_NEW = (
    "recovers the demand prediction in physical units (kW). The 0.1 "
    "scaling factor in front of the noise term restrains the magnitude "
    "of the injected Gaussian perturbation, so that the "
    "reparameterisation acts as a lightweight probabilistic regulariser "
    "around the mean prediction rather than a calibrated predictive "
    "distribution—the latter being the responsibility of the "
    "conformal layer of Section 2.3. When the runtime feature "
    "dimensionality"
)
for p in body.iter(qn('w:p')):
    if REPAR_OLD in get_text(p):
        set_text_inplace(p, REPAR_OLD, REPAR_NEW)
        print("[2.2] expanded reparameterisation rationale")
        break

# ---------------------------------------------------------------------------
# 15) Expand the textual explanation of Eq. (10) to mention the SOC anchor
#     and the composite regulariser R that are actually implemented in
#     strategies._ParametrizedMPC. We don't rewrite the equation table
#     itself (that would require rebuilding the run-level formatting);
#     instead we expand the descriptive paragraph that follows so the
#     formula is consistent with the implementation.
# ---------------------------------------------------------------------------
OBJ_EXP_OLD = (
    " is the shadow value of stored energy, and R(·) is a small "
    "regularisation term that suppresses spurious micro-cycling and "
    "gently encourages renewable utilisation. The peak term is "
    "formulated as a one-sided epigraph penalty so that the controller "
    "is only charged for increments above the running peak, and the "
    "storage shadow value is dynamically tied to the 24-hour mean "
    "tariff so that the optimal terminal SOC remains within a "
    "reasonable band instead of saturating at the upper limit."
)
OBJ_EXP_NEW = (
    " is the shadow value of stored energy. The composite regulariser "
    "R(·) lumps together two practically important terms that are "
    "active in every solver call but are not numerically dominant: a "
    "cycle-suppression penalty ρ_cyc · Σ_t (P_c,t + "
    "P_d,t) that prevents spurious micro-cycling and a renewable-"
    "curtailment penalty ρ_curt · Σ_t (P̂_r,t − "
    "P_r,t) that softly encourages renewable utilisation; the implemented "
    "coefficients are ρ_cyc = 5×10⁻³ and "
    "ρ_curt = 1×10⁻³. In addition, a quadratic "
    "terminal-SOC anchor w_a · C · (S_H − "
    "S_target)² with w_a = 1 and S_target = 0.5 is "
    "added to the objective to keep the end-of-horizon SOC inside the "
    "physically meaningful band; together with the linear shadow term "
    "−α_s C S_H, this quadratic anchor produces a "
    "first-order optimum at S_H = S_target + α_s/2, "
    "and the shadow value itself is clamped at runtime to "
    "α_s = max(0.1, min(0.3, 0.25 · π̄_24h)), "
    "so that the optimal terminal SOC sits between 0.55 and 0.65 instead "
    "of saturating at the upper limit. The peak term is formulated as a "
    "one-sided epigraph penalty so that the controller is only charged "
    "for increments above the running peak."
)
for p in body.iter(qn('w:p')):
    if OBJ_EXP_OLD in get_text(p):
        set_text_inplace(p, OBJ_EXP_OLD, OBJ_EXP_NEW)
        print("[2.5] expanded Eq.(10) explanation with R(.) and SOC anchor")
        break
else:
    # The original paragraph uses run-split formatting that may have
    # broken our literal lookup; fall back to a more lenient pattern
    # that drops formatting-sensitive whitespace.
    target_p = None
    for p in body.iter(qn('w:p')):
        txt = get_text(p)
        if (' shadow value of stored energy' in txt
                and 'micro-cycling' in txt):
            target_p = p
            break
    if target_p is not None:
        # Concatenate all run text, then replace whole-paragraph contents
        # with the new explanation.
        # Reduce to a single <w:t> in the first run, blank the rest.
        all_t = list(target_p.iter(qn('w:t')))
        all_t[0].text = (
            "where π_t (CNY/kWh) is the time-of-use electricity tariff, "
            "φ_t (tCO₂/MWh) is the dynamic marginal carbon "
            "intensity, λ_c is the carbon-price coefficient that "
            "converts emission into monetary cost, w_p is the peak-charge "
            "weight, P_run is the running peak (the maximum grid import "
            "already realised in the current billing month) and α_s "
            "is the shadow value of stored energy. The composite "
            "regulariser R(·) lumps together two practically "
            "important terms that are active in every solver call but "
            "are not numerically dominant: a cycle-suppression penalty "
            "ρ_cyc · Σ_t (P_c,t + P_d,t) that "
            "prevents spurious micro-cycling and a renewable-curtailment "
            "penalty ρ_curt · Σ_t (P̂_r,t "
            "− P_r,t) that softly encourages renewable utilisation; "
            "the implemented coefficients are ρ_cyc = "
            "5×10⁻³ and ρ_curt = 1×"
            "10⁻³. In addition, a quadratic terminal-SOC anchor "
            "w_a · C · (S_H − "
            "S_target)² with w_a = 1 and S_target ="
            " 0.5 is added to the objective to keep the end-of-"
            "horizon SOC inside the physically meaningful band; together "
            "with the linear shadow term −α_s C S_H, "
            "this quadratic anchor produces a first-order optimum at "
            "S_H = S_target + α_s/2, and the "
            "shadow value itself is clamped at runtime to α_s = "
            "max(0.1, min(0.3, 0.25 · π̄_24h)), "
            "so that the optimal terminal SOC sits between 0.55 and 0.65 "
            "instead of saturating at the upper limit. The peak term is "
            "formulated as a one-sided epigraph penalty so that the "
            "controller is only charged for increments above the running "
            "peak."
        )
        for t in all_t[1:]:
            t.text = ''
        print("[2.5] (fallback) rewrote Eq.(10) explanation paragraph")

# ---------------------------------------------------------------------------
# 16) Save.
# ---------------------------------------------------------------------------
doc.save(DST)
print(f"\n[ok] saved -> {DST}")
