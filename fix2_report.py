# -*- coding: utf-8 -*-
"""
fix2_report.py  -  adds TOC + LOT + LOF + LOA + Chiranth name + page breaks
to FISSION_Internship_Report_EDIT.docx
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document("FISSION_Internship_Report_EDIT.docx")
body = doc.element.body

# ── helpers ───────────────────────────────────────────────────────────────────

def mkpara(text, bold=False, size=12, center=False, before=0, after=4, ls=1.5):
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center" if center else "both")
    pPr.append(jc)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), str(int(before * 20)))
    sp.set(qn("w:after"),  str(int(after  * 20)))
    sp.set(qn("w:line"),   str(int(ls * 240)))
    sp.set(qn("w:lineRule"), "auto")
    pPr.append(sp)
    p.append(pPr)
    rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:ascii"), "Times New Roman")
    rf.set(qn("w:hAnsi"), "Times New Roman")
    rPr.append(rf)
    sz  = OxmlElement("w:sz");   sz.set(qn("w:val"),  str(size * 2)); rPr.append(sz)
    szc = OxmlElement("w:szCs"); szc.set(qn("w:val"), str(size * 2)); rPr.append(szc)
    if bold:
        rPr.append(OxmlElement("w:b"))
    r = OxmlElement("w:r"); r.append(rPr)
    t = OxmlElement("w:t"); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t); p.append(r)
    return p

def mkpb():
    p  = OxmlElement("w:p")
    r  = OxmlElement("w:r")
    br = OxmlElement("w:br"); br.set(qn("w:type"), "page")
    r.append(br); p.append(r)
    return p

def has_pagebreak(el):
    return any(n.get(qn("w:type")) == "page" for n in el.iter(qn("w:br")))

def get_text(el):
    return "".join(n.text or "" for n in el.iter(qn("w:t")))

def find_child_idx(keyword, exact=True):
    for i, child in enumerate(list(body)):
        t = get_text(child).strip()
        if (exact and t == keyword) or (not exact and keyword in t):
            return i
    return None

def mktoc_para():
    p = OxmlElement("w:p")
    r1 = OxmlElement("w:r")
    fc1 = OxmlElement("w:fldChar"); fc1.set(qn("w:fldCharType"), "begin")
    r1.append(fc1); p.append(r1)
    r2 = OxmlElement("w:r")
    ins = OxmlElement("w:instrText")
    # Note: Word TOC field switches use single backslash, but we write the literal string
    ins.text = r' TOC \o "1-3" \h \z \u '
    ins.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r2.append(ins); p.append(r2)
    r3 = OxmlElement("w:r")
    fc3 = OxmlElement("w:fldChar"); fc3.set(qn("w:fldCharType"), "end")
    r3.append(fc3); p.append(r3)
    return p

# ── 1. Add Chiranth name after Abhyudaya ──────────────────────────────────────
abhyudaya_idx = find_child_idx("Abhyudaya", exact=False)
if abhyudaya_idx is not None:
    children = list(body)
    # check if next child already has Chiranth
    next_text = get_text(children[abhyudaya_idx + 1]) if abhyudaya_idx + 1 < len(children) else ""
    if "Chiranth" not in next_text:
        chiranth = mkpara("Chiranth D Nandi     [SRN: ___________]",
                          bold=True, size=12, center=True, before=0, after=0, ls=1.0)
        body.insert(abhyudaya_idx + 1, chiranth)
        print("Added Chiranth D Nandi after Abhyudaya M Hegde")
    else:
        print("Chiranth already present")
else:
    print("WARNING: Abhyudaya paragraph not found")

# ── 2. Ensure page break between cover and ACK ────────────────────────────────
ack_idx = find_child_idx("ACKNOWLEDGEMENT", exact=True)
if ack_idx is not None:
    children = list(body)
    if not has_pagebreak(children[ack_idx - 1]):
        body.insert(ack_idx, mkpb())
        print("Page break inserted before ACKNOWLEDGEMENT")
    else:
        print("Page break already present before ACKNOWLEDGEMENT")

# ── 3. Ensure page break between ACK and ABSTRACT ────────────────────────────
abs_idx = find_child_idx("ABSTRACT", exact=True)
if abs_idx is not None:
    children = list(body)
    if not has_pagebreak(children[abs_idx - 1]):
        body.insert(abs_idx, mkpb())
        print("Page break inserted before ABSTRACT")
    else:
        print("Page break already present before ABSTRACT")

# ── 4. Add TOC + LOT + LOF + LOA before CHAPTER 1 ────────────────────────────
ch1_idx = find_child_idx("CHAPTER 1", exact=True)
if ch1_idx is None:
    # try non-exact
    ch1_idx = find_child_idx("CHAPTER 1", exact=False)

if ch1_idx is not None:
    print(f"CHAPTER 1 found at body child index {ch1_idx}")

    new_els = []

    # ── TABLE OF CONTENTS ──
    new_els.append(mkpb())
    new_els.append(mkpara("TABLE OF CONTENTS", bold=True, size=16, center=True, before=0, after=8, ls=1.0))
    new_els.append(mktoc_para())
    new_els.append(mkpara(
        "[Right-click the field above and select Update Field to generate the Table of Contents.]",
        bold=False, size=10, center=False, before=4, after=0, ls=1.0
    ))

    # ── LIST OF TABLES ──
    new_els.append(mkpb())
    new_els.append(mkpara("LIST OF TABLES", bold=True, size=16, center=True, before=0, after=8, ls=1.0))
    for tno, ttitle in [
        ("Table 2.1", "Attack layers, adversary capability, target stage, and success metric"),
        ("Table 3.1", "Positioning of FISSION against representative related work"),
        ("Table 4.1", "Metric definitions used throughout the benchmark"),
        ("Table 5.1", "Registered tools and their schemas (the attack targets of the pipeline)"),
        ("Table 5.2", "Model profiles currently registered (excerpt)"),
        ("Table 6.1", "Attack modules (all implemented and runnable)"),
        ("Table 6.2", "Defence modules"),
        ("Table 6.3", "Test suites and what they pin down"),
        ("Table 7.1", "Committed R1 benchmark configuration (LLaVA, FGSM)"),
        ("Table 7.2", "R1 benchmark results against LLaVA (mean over 5 images; from committed manifests)"),
        ("Table 7.3", "Defence-side validation on the delivered suites"),
        ("Table 7.4", "Cross-architecture benchmark status (honest snapshot)"),
    ]:
        new_els.append(mkpara(f"{tno}  .....  {ttitle}", size=11, before=0, after=3, ls=1.0))

    # ── LIST OF FIGURES ──
    new_els.append(mkpb())
    new_els.append(mkpara("LIST OF FIGURES", bold=True, size=16, center=True, before=0, after=8, ls=1.0))
    for fno, ftitle in [
        ("Figure 4.1", "Attack layers R0-R5b mapped to the pipeline stage each one compromises"),
        ("Figure 5.1", "FISSION system architecture with attack surfaces and defence placement"),
        ("Figure 5.2", "Docker deployment: eight services on one bridge network"),
        ("Figure 7.1", "Mean ISR and SSIM across epsilon. SBI holds only at epsilon = 2/255"),
        ("Figure 7.2", "Per-image ISR showing document-dependent susceptibility"),
        ("Figure 7.3", "Mean SDS by epsilon; all cells lie above the 0.30 drift threshold"),
    ]:
        new_els.append(mkpara(f"{fno}  .....  {ftitle}", size=11, before=0, after=3, ls=1.0))

    # ── LIST OF ABBREVIATIONS ──
    new_els.append(mkpb())
    new_els.append(mkpara("LIST OF ABBREVIATIONS", bold=True, size=16, center=True, before=0, after=8, ls=1.0))
    for abbr, exp in [
        ("ADR",    "Attention Dominance Ratio"),
        ("AER",    "Attention Entropy Ratio"),
        ("APR",    "Agent Propagation Rate"),
        ("ASR",    "Attack Success Rate"),
        ("ASV",    "Argument Semantic Validation"),
        ("BSD",    "Baseline Score Drift"),
        ("CA",     "Clean Accuracy"),
        ("CBD",    "Context Backdoor Depth"),
        ("CLIP",   "Contrastive Language-Image Pre-Training"),
        ("CLPR",   "Cross-Layer Pearson coefficient (epsilon vs ISR correlation)"),
        ("CPR",    "Context Poison Rate"),
        ("FGSM",   "Fast Gradient Sign Method"),
        ("FPR",    "False Positive Rate"),
        ("HMAC",   "Hash-based Message Authentication Code"),
        ("ICS",    "Intent-Call Similarity"),
        ("ISR",    "Injection Success Rate"),
        ("KL",     "Kullback-Leibler divergence"),
        ("MAD",    "Median Absolute Deviation"),
        ("MCP",    "Model Context Protocol"),
        ("PGD",    "Projected Gradient Descent"),
        ("SBI",    "Stealthiness-Benchmark Index (ISR >= 0.50 AND SSIM >= 0.95)"),
        ("SDS",    "Semantic Drift Score"),
        ("SHA256", "Secure Hash Algorithm 256-bit"),
        ("SSIM",   "Structural Similarity Index Measure"),
        ("TCIV",   "Tool Call Integrity Verifier"),
        ("TCR",    "Tool-Call Corruption Rate"),
        ("TDR",    "Trojan Detection Rate"),
        ("VLM",    "Vision-Language Model"),
    ]:
        new_els.append(mkpara(f"{abbr:<8}  {exp}", size=11, before=0, after=3, ls=1.0))

    new_els.append(mkpb())

    # Insert all elements before CHAPTER 1
    ch1_idx = find_child_idx("CHAPTER 1", exact=True)  # re-scan after inserts
    for el in reversed(new_els):
        body.insert(ch1_idx, el)
    print(f"Inserted {len(new_els)} elements (TOC + LOT + LOF + LOA) before CHAPTER 1")
else:
    print("WARNING: CHAPTER 1 not found — TOC/LOT/LOF not inserted")

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save("FISSION_Internship_Report_EDIT.docx")
print("\nDONE - saved FISSION_Internship_Report_EDIT.docx")
print("Open in Word -> press Ctrl+A -> F9 to update TOC field.")
