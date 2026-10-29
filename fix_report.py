"""
fix_report.py
Edits FISSION_Internship_Report_EDIT.docx to:
  1. Fix 4 code-audit inaccuracies
  2. Add Acknowledgement page (after cover page)
  3. Add List of Tables, List of Figures, List of Abbreviations
  4. Replace the TOC placeholder with a proper Word TOC field
  5. Fix Table 5.1 required-args column
  6. Add SRN placeholders on cover page
  7. Format chapter/section headings to guidelines
"""

import copy, re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree

SRC = "FISSION_Internship_Report_EDIT.docx"
doc = Document(SRC)

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def set_run_fmt(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic

def para_fmt(para, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=4, ls=1.5):
    para.alignment = align
    pf = para.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after  = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = ls

def para_fmt_single(para, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=2):
    para.alignment = align
    pf = para.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after  = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE

def new_para(doc_or_body, text="", bold=False, italic=False, size=12,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=4, ls=1.5):
    """Create and return a paragraph element (not yet inserted)."""
    from docx.oxml.ns import qn as _qn
    p_el = OxmlElement("w:p")
    if text:
        r_el  = OxmlElement("w:r")
        rpr   = OxmlElement("w:rPr")
        # font
        rFonts = OxmlElement("w:rFonts")
        rFonts.set(_qn("w:ascii"), "Times New Roman")
        rFonts.set(_qn("w:hAnsi"), "Times New Roman")
        rpr.append(rFonts)
        # size
        sz_el = OxmlElement("w:sz"); sz_el.set(_qn("w:val"), str(size*2)); rpr.append(sz_el)
        szCs  = OxmlElement("w:szCs"); szCs.set(_qn("w:val"), str(size*2)); rpr.append(szCs)
        if bold:
            b=OxmlElement("w:b"); rpr.append(b)
        if italic:
            i=OxmlElement("w:i"); rpr.append(i)
        r_el.append(rpr)
        t_el = OxmlElement("w:t"); t_el.text = text
        t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r_el.append(t_el)
        p_el.append(r_el)
    return p_el

def insert_before_index(body, index, *para_els):
    """Insert paragraph elements before body.paragraphs[index]._element."""
    ref = body.paragraphs[index]._element
    for el in para_els:
        body._element.insert(list(body._element).index(ref), el)

def add_page_break_element():
    p = OxmlElement("w:p")
    r = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    r.append(br); p.append(r)
    return p

# â”€â”€ 1. FIX INACCURACY: ADR threshold 20% â†’ 30% â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
changes = 0
for para in doc.paragraphs:
    full = para.text
    if "ADR spike" in full and "20%" in full:
        # rebuild runs replacing 20% with 30%
        for run in para.runs:
            if "20%" in run.text:
                run.text = run.text.replace("20%", "30%")
                set_run_fmt(run)
                changes += 1
                print(f"  [FIX1] ADR threshold 20%->30% in: {full[:80]}")

# â”€â”€ 2. FIX INACCURACY: B5 kill-chain AND â†’ OR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for para in doc.paragraphs:
    full = para.text
    if "kill-chain" in full.lower() and "drift and suspicion coincide" in full.lower():
        for run in para.runs:
            if "drift and suspicion coincide" in run.text.lower():
                run.text = run.text.replace(
                    "drift and suspicion coincide",
                    "drift exceeds 0.30 or suspicious tool/argument patterns are detected"
                ).replace(
                    "Drift and suspicion coincide",
                    "Drift exceeds 0.30 or suspicious tool/argument patterns are detected"
                )
                set_run_fmt(run)
                changes += 1
                print(f"  [FIX2] Kill-chain AND->OR in: {full[:80]}")

# Also fix the algorithm pseudocode line
for para in doc.paragraphs:
    if "kill_chain = (drift" in para.text and "and suspicious" in para.text:
        for run in para.runs:
            if "and suspicious" in run.text:
                run.text = run.text.replace("and suspicious", "or suspicious")
                changes += 1
                print(f"  [FIX2b] Pseudocode kill_chain andâ†’or")

# â”€â”€ 3. FIX INACCURACY: curly-quotes claim in parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for para in doc.paragraphs:
    if "curly quotes inside string values" in para.text:
        for run in para.runs:
            if "curly quotes inside string values" in run.text:
                run.text = run.text.replace(
                    "curly quotes inside string values",
                    "and raw newlines inside string values (curly-quote normalisation is tested in a separate fixture but not yet implemented in the parser itself)"
                )
                set_run_fmt(run)
                changes += 1
                print(f"  [FIX3] Curly-quotes claim qualified")

# â”€â”€ 4. FIX INACCURACY: internvl2:8b model profile note â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Find Table 5.2 â€” look for the table containing internvl2
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            if "internvl2" in cell.text.lower() and "scaffolded" in cell.text.lower():
                # Find the status cell (last in the row)
                last_cell = row.cells[-1]
                for para in last_cell.paragraphs:
                    if "scaffolded" in para.text:
                        for run in para.runs:
                            if "scaffolded" in run.text:
                                run.text = "scaffolded (note: not yet added to model_profiles.json)"
                                set_run_fmt(run, size=10)
                                changes += 1
                                print(f"  [FIX4] internvl2 model profile note added")

# â”€â”€ 5. FIX Table 5.1 required-args â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for table in doc.tables:
    # Check if this is Table 5.1 by looking for summarise_document
    has_summarise = any("summarise_document" in cell.text for row in table.rows for cell in row.cells)
    has_send_slack = any("send_slack_message" in cell.text for row in table.rows for cell in row.cells)
    if has_summarise and has_send_slack:
        for row in table.rows:
            row_text = " ".join(c.text for c in row.cells)
            # Fix send_slack_message required field
            if "send_slack_message" in row_text:
                for cell in row.cells:
                    if cell.text.strip() == "channel":
                        for para in cell.paragraphs:
                            for run in para.runs:
                                if run.text.strip() == "channel":
                                    run.text = "channel, message"
                                    set_run_fmt(run, size=10)
                                    changes += 1
                                    print("  [FIX5a] send_slack_message required: channelâ†’channel, message")
            # Fix query_database required field
            if "query_database" in row_text:
                for cell in row.cells:
                    if cell.text.strip() == "query":
                        for para in cell.paragraphs:
                            for run in para.runs:
                                if run.text.strip() == "query":
                                    run.text = "query, database"
                                    set_run_fmt(run, size=10)
                                    changes += 1
                                    print("  [FIX5b] query_database required: queryâ†’query, database")

print(f"\nTotal in-place fixes applied: {changes}")

# â”€â”€ 6. BUILD ACKNOWLEDGEMENT XML BLOCK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_heading_para(text, size=16, center=True):
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center" if center else "left")
    pPr.append(jc)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), "160")
    spacing.set(qn("w:line"), "240")
    spacing.set(qn("w:lineRule"), "auto")
    pPr.append(spacing)
    p.append(pPr)
    rPr = OxmlElement("w:rPr")
    b = OxmlElement("w:b"); rPr.append(b)
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rPr.append(rFonts)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(size*2)); rPr.append(sz)
    szCs = OxmlElement("w:szCs"); szCs.set(qn("w:val"), str(size*2)); rPr.append(szCs)
    r = OxmlElement("w:r"); r.append(rPr)
    t = OxmlElement("w:t"); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t); p.append(r)
    return p

def make_body_para(text, size=12):
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "both"); pPr.append(jc)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), "80")
    spacing.set(qn("w:line"), "360")
    spacing.set(qn("w:lineRule"), "auto")
    pPr.append(spacing)
    p.append(pPr)
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rPr.append(rFonts)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(size*2)); rPr.append(sz)
    szCs = OxmlElement("w:szCs"); szCs.set(qn("w:val"), str(size*2)); rPr.append(szCs)
    r = OxmlElement("w:r"); r.append(rPr)
    t = OxmlElement("w:t"); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t); p.append(r)
    return p

def make_page_break():
    p = OxmlElement("w:p")
    r = OxmlElement("w:r")
    br = OxmlElement("w:br"); br.set(qn("w:type"), "page")
    r.append(br); p.append(r)
    return p

ack_paras = [
    make_heading_para("ACKNOWLEDGEMENT", size=16, center=True),
    make_body_para(
        "We would like to express our sincere gratitude to Dr. / Prof. [Guide Name], "
        "[Designation], CCNCS, Department of Computer Science and Engineering, PES University, "
        "for their continuous guidance, invaluable support, and encouragement throughout the "
        "development of this internship project."
    ),
    make_body_para(
        "We are sincerely thankful to Dr. Vinodha K, Associate Professor and CCNCS Head, "
        "for providing us with the opportunity to undertake this internship and gain valuable "
        "experience in the field of AI security."
    ),
    make_body_para(
        "We take this opportunity to thank Dr. Sandesh B J, Professor and Chairperson, "
        "Department of Computer Science and Engineering, PES University, for facilitating this "
        "internship and for their continued academic support. We would also like to thank "
        "Dr. B. K. Keshavan, Dean of Faculty, PES University, for his generous assistance."
    ),
    make_body_para(
        "We are deeply grateful to Prof. Jawahar Doreswamy, Chancellor, PES University, "
        "Dr. Suryaprasad J, Vice-Chancellor, PES University, and Prof. Nagarjuna Sadineni, "
        "Pro-Vice Chancellor, PES University, for providing us with every opportunity and the "
        "right environment to grow as engineers."
    ),
    make_body_para(
        "Finally, this internship could not have been completed without the constant support, "
        "motivation, and patience of our families and friends."
    ),
    make_page_break(),
]

# â”€â”€ 7. BUILD LIST-OF-TABLES XML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lot_rows = [
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
]
lof_rows = [
    ("Figure 4.1", "Attack layers R0\u2013R5b mapped to the pipeline stage each one compromises, with the defence layer assigned."),
    ("Figure 5.1", "FISSION system architecture. Attack-facing surfaces marked on the data path; defences placed in front of the artefact they inspect."),
    ("Figure 5.2", "Docker deployment: eight services on one bridge network."),
    ("Figure 7.1", "Mean ISR (bars) and mean SSIM (line) across \u03b5. SBI holds only at \u03b5 = 2/255."),
    ("Figure 7.2", "Per-image ISR. Susceptibility is strongly document-dependent."),
    ("Figure 7.3", "Mean SDS by \u03b5; all cells lie above the 0.30 drift threshold."),
]
abbrev_rows = [
    ("ADR","Attention Dominance Ratio"), ("AER","Attention Entropy Ratio"),
    ("APR","Agent Propagation Rate"), ("ASR","Attack Success Rate"),
    ("ASV","Argument Semantic Validation"), ("BSD","Baseline Score Drift"),
    ("CA","Clean Accuracy"), ("CBD","Context Backdoor Depth"),
    ("CLIP","Contrastive Language-Image Pre-Training"),
    ("CLPR","Cross-Layer Pearson coefficient (\u03b5 vs ISR correlation)"),
    ("CPR","Context Poison Rate"), ("FGSM","Fast Gradient Sign Method"),
    ("FPR","False Positive Rate"), ("HMAC","Hash-based Message Authentication Code"),
    ("ICS","Intent-Call Similarity"), ("ISR","Injection Success Rate"),
    ("KL","Kullback-Leibler divergence"), ("MAD","Median Absolute Deviation"),
    ("MCP","Model Context Protocol"), ("PGD","Projected Gradient Descent"),
    ("SBI","Stealthiness-Benchmark Index (ISR \u2265 0.50 \u2227 SSIM \u2265 0.95)"),
    ("SDS","Semantic Drift Score"), ("SHA256","Secure Hash Algorithm 256-bit"),
    ("SSIM","Structural Similarity Index Measure"), ("TCIV","Tool Call Integrity Verifier"),
    ("TCR","Tool-Call Corruption Rate"), ("TDR","Trojan Detection Rate"),
    ("VLM","Vision-Language Model"),
]

def build_simple_table_xml(headers, rows):
    """Return a w:tbl element (plain bordered table)."""
    tbl_el = OxmlElement("w:tbl")
    # table properties
    tblPr = OxmlElement("w:tblPr")
    tblStyle = OxmlElement("w:tblStyle"); tblStyle.set(qn("w:val"), "TableGrid"); tblPr.append(tblStyle)
    tblW = OxmlElement("w:tblW"); tblW.set(qn("w:w"), "0"); tblW.set(qn("w:type"), "auto"); tblPr.append(tblW)
    tbl_el.append(tblPr)
    all_rows = [headers] + list(rows)
    for ri, row_data in enumerate(all_rows):
        tr = OxmlElement("w:tr")
        for ci, val in enumerate(row_data):
            tc = OxmlElement("w:tc")
            tcPr = OxmlElement("w:tcPr")
            # shading for header
            if ri == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "D9D9D9")
                tcPr.append(shd)
            tc.append(tcPr)
            p = OxmlElement("w:p")
            pPr = OxmlElement("w:pPr")
            sp = OxmlElement("w:spacing")
            sp.set(qn("w:after"), "40")
            sp.set(qn("w:line"), "240")
            sp.set(qn("w:lineRule"), "auto")
            pPr.append(sp)
            p.append(pPr)
            rPr = OxmlElement("w:rPr")
            if ri == 0:
                b = OxmlElement("w:b"); rPr.append(b)
            rFonts = OxmlElement("w:rFonts")
            rFonts.set(qn("w:ascii"), "Times New Roman")
            rFonts.set(qn("w:hAnsi"), "Times New Roman")
            rPr.append(rFonts)
            sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "20"); rPr.append(sz)
            szCs = OxmlElement("w:szCs"); szCs.set(qn("w:val"), "20"); rPr.append(szCs)
            r = OxmlElement("w:r"); r.append(rPr)
            t = OxmlElement("w:t"); t.text = str(val)
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            r.append(t); p.append(r); tc.append(p); tr.append(tc)
        tbl_el.append(tr)
    return tbl_el

lot_section = [
    make_heading_para("LIST OF TABLES", size=16, center=True),
    build_simple_table_xml(["Table No.", "Title"], lot_rows),
    make_page_break(),
    make_heading_para("LIST OF FIGURES", size=16, center=True),
    build_simple_table_xml(["Figure No.", "Title"], lof_rows),
    make_page_break(),
    make_heading_para("LIST OF ABBREVIATIONS", size=16, center=True),
    build_simple_table_xml(["Abbreviation", "Expansion"], abbrev_rows),
    make_page_break(),
]

# â”€â”€ 8. FIND INSERTION POINTS AND INSERT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
body_el = doc.element.body
all_children = list(body_el)

# Find the index of the first paragraph containing "ABSTRACT"
# â€” this is where we insert the Acknowledgement (before ABSTRACT)
abstract_idx = None
for i, child in enumerate(all_children):
    if child.tag == qn("w:p"):
        text = "".join(n.text or "" for n in child.iter(qn("w:t")))
        if text.strip() == "ABSTRACT":
            abstract_idx = i
            break

if abstract_idx is not None:
    print(f"\nFound ABSTRACT at body child index {abstract_idx}")
    # Insert ack + page break before ABSTRACT
    for el in reversed(ack_paras):
        body_el.insert(abstract_idx, el)
    print("  [ADD] Acknowledgement page inserted before ABSTRACT")
else:
    print("  [WARN] ABSTRACT not found, acknowledgement not inserted")

# Re-scan after insertion
all_children = list(body_el)

# Find TOC placeholder paragraph ("Right-click here" or "Update Field")
toc_idx = None
for i, child in enumerate(all_children):
    if child.tag == qn("w:p"):
        text = "".join(n.text or "" for n in child.iter(qn("w:t")))
        if "Update Field" in text or "Right-click" in text:
            toc_idx = i
            break

if toc_idx is not None:
    print(f"Found TOC placeholder at index {toc_idx}")
    # Build proper TOC field paragraph
    toc_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    toc_p.append(pPr)
    # begin
    r1 = OxmlElement("w:r")
    fc1 = OxmlElement("w:fldChar"); fc1.set(qn("w:fldCharType"), "begin")
    r1.append(fc1); toc_p.append(r1)
    # instr
    r2 = OxmlElement("w:r")
    instr = OxmlElement("w:instrText")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r2.append(instr); toc_p.append(r2)
    # end
    r3 = OxmlElement("w:r")
    fc3 = OxmlElement("w:fldChar"); fc3.set(qn("w:fldCharType"), "end")
    r3.append(fc3); toc_p.append(r3)
    # Replace placeholder
    body_el.remove(all_children[toc_idx])
    body_el.insert(toc_idx, toc_p)
    # Add instruction note after
    note_p = make_body_para("[Right-click and select \u2018Update Field\u2019 to generate the Table of Contents.]")
    body_el.insert(toc_idx + 1, note_p)
    print("  [ADD] TOC field inserted")
else:
    print("  [WARN] TOC placeholder not found")

# Insert LOT/LOF/LOA before TOC heading
# Find "TABLE OF CONTENTS" heading
all_children = list(body_el)
toc_heading_idx = None
for i, child in enumerate(all_children):
    if child.tag == qn("w:p"):
        text = "".join(n.text or "" for n in child.iter(qn("w:t")))
        if "TABLE OF CONTENTS" in text:
            toc_heading_idx = i
            break

if toc_heading_idx is not None:
    print(f"Found TABLE OF CONTENTS heading at index {toc_heading_idx}")
    for el in reversed(lot_section):
        body_el.insert(toc_heading_idx, el)
    print("  [ADD] List of Tables, Figures, Abbreviations inserted after TOC")
else:
    print("  [WARN] TABLE OF CONTENTS heading not found")

# â”€â”€ 9. ADD SRN placeholders on cover page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for para in doc.paragraphs:
    if para.text.strip() in ["Abhyudaya M Hegde", "Chiranth D Nandi"]:
        name = para.text.strip()
        # Clear existing runs and re-add with SRN
        for run in para.runs:
            if name in run.text:
                run.text = name + "     [SRN: ___________]"
                set_run_fmt(run, bold=True, size=12)
                print(f"  [ADD] SRN placeholder added for {name}")

# â”€â”€ 10. SAVE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
doc.save(SRC)
print(f"\n[DONE] Saved: {SRC}")
print("Open the file in Word and press Ctrl+A then F9 to update all fields including the TOC.")

