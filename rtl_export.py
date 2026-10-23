"""
rtl_export.py - direction-aware export helpers for OCRAIHub (stdlib only, no Qt).

Why this exists
---------------
The Qt text widget picks a base direction per paragraph, so Arabic looks right on
screen. Markdown and Word do not: a .md file has no direction info, and Word stores
direction as a paragraph property (<w:bidi/>). This module reproduces the Qt behaviour
in the exported files.

Public API
----------
detect_direction(text, fallback=None) -> "rtl" | "ltr" | None
tidy_arabic_punctuation(text)         -> str      (optional OCR cleanup)
prepare_markdown(md, mode)            -> str      mode: "none" | "rlm" | "html"
markdown_to_html(md, full_page=False) -> str      <p dir="auto">-style output, clipboard-friendly
export_docx(md, path, **options)      -> None     real RTL .docx, no third-party packages

Supported markdown subset (what OCR output actually contains):
headings (#..######), paragraphs, **bold**, *italic*, pipe tables, and lines of
3+ dashes (treated as page separators).
"""
from __future__ import annotations

import html
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

RLM = "\u200f"

# --------------------------------------------------------------------------- #
# Direction detection
# --------------------------------------------------------------------------- #

def _strong_class(ch: str) -> Optional[str]:
    """'rtl' for R/AL, 'ltr' for L, None for everything else (digits, punctuation...)."""
    b = unicodedata.bidirectional(ch)
    if b in ("R", "AL"):
        return "rtl"
    if b == "L":
        return "ltr"
    return None


def detect_direction(text: str, fallback: Optional[str] = None) -> Optional[str]:
    """First-strong-character rule (same idea as Qt / UBA rule P2-P3).

    Returns "rtl" or "ltr"; returns `fallback` when the text has no strong
    character (e.g. a bare page number such as '١٣٦').
    """
    for ch in text:
        c = _strong_class(ch)
        if c:
            return c
    return fallback


# --------------------------------------------------------------------------- #
# Optional OCR text cleanup
# --------------------------------------------------------------------------- #

_SPACE_BEFORE_PUNCT = re.compile(r"(?<=[\u0600-\u06FF])[ \t]+([،؛؟.:!])")


_UI_HEADER = re.compile(r"^\s*--- OCR Result .*---\s*$")
_UI_SEPARATOR = re.compile(r"^\s*-{40}\s*$")


def strip_ui_decorations(text: str) -> str:
    """Remove the labels the app adds to its on-screen log of results.

    The result box contains '--- OCR Result (... Batch 1/3) ---' headers and a
    40-dash line after every batch. They are UI scaffolding, not document
    content, so they are dropped before exporting to Word / RTL Markdown.
    """
    kept = [ln for ln in text.replace("\r\n", "\n").split("\n")
            if not _UI_HEADER.match(ln) and not _UI_SEPARATOR.match(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n") + "\n"


def tidy_arabic_punctuation(text: str) -> str:
    """Remove the stray space OCR leaves between an Arabic word and ، ؛ ؟ . : !

    Conservative on purpose: only touches a space that sits directly after an
    Arabic-script letter, so tables, markdown rules and Latin text are untouched.
    """
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


# --------------------------------------------------------------------------- #
# Markdown parsing (tiny subset)
# --------------------------------------------------------------------------- #

@dataclass
class Block:
    kind: str                      # "heading" | "para" | "table" | "pagebreak"
    text: str = ""
    level: int = 0
    rows: List[List[str]] = field(default_factory=list)


_HR = re.compile(r"^\s*-{3,}\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_markdown(md: str) -> List[Block]:
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: List[Block] = []
    para: List[str] = []

    def flush():
        if para:
            blocks.append(Block("para", "\n".join(para)))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
        elif _HR.match(line):
            flush()
            blocks.append(Block("pagebreak"))
        elif _HEADING.match(line):
            flush()
            m = _HEADING.match(line)
            blocks.append(Block("heading", m.group(2), level=len(m.group(1))))
        elif "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(Block("table", rows=rows))
            continue
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)")


def _inline_spans(text: str) -> List[Tuple[str, bool, bool]]:
    """Split into (text, bold, italic) spans."""
    out: List[Tuple[str, bool, bool]] = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((part[1:-1], False, True))
        else:
            out.append((part, False, False))
    return out


# --------------------------------------------------------------------------- #
# Markdown output modes
# --------------------------------------------------------------------------- #

def prepare_markdown(md: str, mode: str = "html") -> str:
    """Make a .md file carry direction information.

    mode="none": unchanged.
    mode="rlm" : prefix every RTL paragraph with U+200F (invisible). Helps plain
                 editors that implement the bidi algorithm; does NOT help Word.
    mode="html": wrap each RTL paragraph/heading in <div dir="rtl">. Works in
                 Typora, Obsidian, GitHub, VS Code preview and most renderers.
    """
    if mode == "none":
        return md
    if mode not in ("rlm", "html"):
        raise ValueError("mode must be 'none', 'rlm' or 'html'")

    out: List[str] = []
    prev = None
    for blk in _split_keep_layout(md):
        kind, text = blk
        if kind != "text":
            out.append(text)
            continue
        d = detect_direction(text, fallback=prev)
        if d:
            prev = d
        if d == "rtl":
            if mode == "rlm":
                out.append("\n".join(RLM + ln if ln.strip() else ln for ln in text.split("\n")))
            else:
                out.append(f'<div dir="rtl">\n\n{text}\n\n</div>')
        else:
            out.append(text)
    return "\n\n".join(out)


def _split_keep_layout(md: str) -> List[Tuple[str, str]]:
    """Split on blank lines; tables, code fences and rules are 'raw' (left as-is)."""
    parts = re.split(r"\n\s*\n", md.replace("\r\n", "\n").strip("\n"))
    res: List[Tuple[str, str]] = []
    for p in parts:
        s = p.strip("\n")
        if not s.strip():
            continue
        lines = s.split("\n")
        first = lines[0]
        is_table = "|" in first and len(lines) > 1 and bool(_TABLE_SEP.match(lines[1]))
        is_raw = bool(_HR.match(first)) or first.lstrip().startswith("```") or is_table
        res.append(("raw" if is_raw else "text", s))
    return res


# --------------------------------------------------------------------------- #
# HTML (screen parity + rich clipboard for Word)
# --------------------------------------------------------------------------- #

def _inline_html(text: str) -> str:
    parts = []
    for t, b, i in _inline_spans(text):
        e = html.escape(t)
        if b:
            e = f"<b>{e}</b>"
        if i:
            e = f"<i>{e}</i>"
        parts.append(e)
    return "".join(parts)


def markdown_to_html(md: str, full_page: bool = False, font: str = "Arial", size_pt: float = 14) -> str:
    """HTML with an explicit dir per paragraph.

    Put this on the clipboard as text/html (QMimeData.setHtml) and Word keeps RTL
    on paste. Explicit dir="rtl"/"ltr" is used instead of dir="auto" because Word
    ignores "auto".
    """
    blocks = parse_markdown(md)
    prev = "ltr"
    body: List[str] = []
    style = f"font-family:'{font}';font-size:{size_pt}pt;"
    for b in blocks:
        if b.kind == "pagebreak":
            body.append('<br clear="all" style="page-break-before:always">')
        elif b.kind in ("para", "heading"):
            d = detect_direction(b.text, fallback=prev) or prev
            prev = d
            align = "right" if d == "rtl" else "left"
            inner = "<br>".join(_inline_html(ln) for ln in b.text.split("\n"))
            tag = f"h{min(b.level, 6)}" if b.kind == "heading" else "p"
            body.append(
                f'<{tag} dir="{d}" style="direction:{d};unicode-bidi:embed;text-align:{align};{style}">{inner}</{tag}>'
            )
        elif b.kind == "table":
            tdir = detect_direction(" ".join(" ".join(r) for r in b.rows), fallback=prev) or prev
            rows_html = []
            for ri, row in enumerate(b.rows):
                cell_tag = "th" if ri == 0 else "td"
                cells = []
                for c in row:
                    cd = detect_direction(c, fallback=tdir) or tdir
                    cells.append(f'<{cell_tag} dir="{cd}" style="{style}">{_inline_html(c)}</{cell_tag}>')
                rows_html.append("<tr>" + "".join(cells) + "</tr>")
            body.append(f'<table dir="{tdir}" border="1" style="border-collapse:collapse">' + "".join(rows_html) + "</table>")
    content = "\n".join(body)
    if full_page:
        return f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"></head><body>\n{content}\n</body></html>'
    return content


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _x(s: str) -> str:
    """XML-escape, and drop C0 control characters Word refuses to open.

    OCR engines occasionally emit these for a scan artifact or ink smudge.
    They're invisible in plain text but make Word show "unreadable content"
    and refuse to open the file. \t \n \r are NOT in this range and pass
    through untouched (newlines are handled separately as <w:br/> anyway).
    """
    s = _CONTROL_CHAR_RE.sub("", s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class DocxOptions:
    latin_font: str = "Arial"
    arabic_font: str = "Arial"
    size_pt: float = 14
    heading_scale: Tuple[float, ...] = (1.6, 1.4, 1.2, 1.1, 1.0, 1.0)
    page_break_on_rule: bool = True
    # Direction for paragraphs with no strong character (bare page numbers etc.):
    # "document" = majority direction of the whole file, "previous" = same as the
    # paragraph above, or force "rtl" / "ltr".
    neutral_direction: str = "document"
    tidy_punctuation: bool = False


def _run_xml(text: str, rtl: bool, bold: bool, italic: bool, o: DocxOptions, size_mult: float = 1.0) -> str:
    half = int(round(o.size_pt * size_mult * 2))
    rpr = f'<w:rFonts w:ascii="{o.latin_font}" w:hAnsi="{o.latin_font}" w:cs="{o.arabic_font}"/>'
    if bold:
        rpr += "<w:b/><w:bCs/>"
    if italic:
        rpr += "<w:i/><w:iCs/>"
    rpr += f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/>'
    if rtl:
        rpr += "<w:rtl/>"
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{_x(text)}</w:t></w:r>'


def _para_xml(text: str, direction: str, o: DocxOptions, style: Optional[str] = None,
              size_mult: float = 1.0, force_bold: bool = False, in_table: bool = False) -> str:
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if direction == "rtl":
        ppr += "<w:bidi/>"
    # In a bidi paragraph, jc "left" means the *start* edge, i.e. right-aligned.
    ppr += '<w:jc w:val="left"/>' if not in_table or direction == "rtl" else ""
    ppr += "<w:spacing w:after=\"120\"/>" if not in_table else ""
    runs: List[str] = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        if li:
            runs.append("<w:r><w:br/></w:r>")
        # IMPORTANT: Word decides where spaces, commas, periods, parentheses and
        # digits go from the run's <w:rtl/> flag. Splitting a paragraph into
        # separate Arabic / Latin runs puts those neutrals into LTR runs and
        # they jump to the wrong side. So: every run in an RTL paragraph is
        # flagged rtl (Latin words still read left-to-right, by their own
        # strong direction), and no run in an LTR paragraph is.
        for span, bold, italic in _inline_spans(line):
            runs.append(_run_xml(span, direction == "rtl", bold or force_bold, italic, o, size_mult))
    return f"<w:p><w:pPr>{ppr}</w:pPr>{''.join(runs)}</w:p>"


def _table_xml(rows: List[List[str]], direction: str, o: DocxOptions) -> str:
    rows = [r for r in rows if r]          # drop stray empty rows (e.g. a bare "|" line)
    if not rows:
        return ""                          # nothing usable; caller skips this block
    ncols = max(len(r) for r in rows)
    if ncols == 0:
        return ""
    total = 9000
    w = total // ncols
    tblpr = (
        f'<w:tblW w:w="{total}" w:type="dxa"/>'
        + ("<w:bidiVisual/>" if direction == "rtl" else "")
        + '<w:tblBorders>'
        + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                  for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders>"
        + '<w:tblLayout w:type="fixed"/>'
        + '<w:tblCellMar><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>'
    )
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for _ in range(ncols))
    trs = []
    for ri, row in enumerate(rows):
        row = row + [""] * (ncols - len(row))
        tcs = []
        for c in row:
            cd = detect_direction(c, fallback=direction) or direction
            tcs.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>'
                f'{_para_xml(c, cd, o, force_bold=(ri == 0), in_table=True)}</w:tc>'
            )
        trs.append("<w:tr>" + "".join(tcs) + "</w:tr>")
    return f"<w:tbl><w:tblPr>{tblpr}</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"


def _styles_xml(o: DocxOptions) -> str:
    half = int(round(o.size_pt * 2))

    def heading(n: int) -> str:
        mult = o.heading_scale[n - 1]
        return (
            f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/>'
            f'<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>'
            f'<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="{n-1}"/></w:pPr>'
            f'<w:rPr><w:b/><w:bCs/><w:sz w:val="{int(round(o.size_pt*mult*2))}"/>'
            f'<w:szCs w:val="{int(round(o.size_pt*mult*2))}"/></w:rPr></w:style>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults><w:rPrDefault><w:rPr>'
        f'<w:rFonts w:ascii="{o.latin_font}" w:hAnsi="{o.latin_font}" w:cs="{o.arabic_font}"/>'
        f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/>'
        '<w:lang w:val="en-US" w:bidi="ar-SA"/></w:rPr></w:rPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
        + "".join(heading(n) for n in range(1, 7))
        + "</w:styles>"
    )


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)
_DOC_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)


def build_document_xml(md: str, o: Optional[DocxOptions] = None) -> str:
    o = o or DocxOptions()
    if o.tidy_punctuation:
        md = tidy_arabic_punctuation(md)
    body: List[str] = []
    blocks = parse_markdown(md)
    votes = {"rtl": 0, "ltr": 0}
    for b in blocks:
        if b.kind in ("para", "heading"):
            d0 = detect_direction(b.text)
            if d0:
                votes[d0] += 1
    dominant = "rtl" if votes["rtl"] >= votes["ltr"] else "ltr"
    prev = dominant
    for b in blocks:
        if b.kind == "pagebreak":
            if o.page_break_on_rule:
                body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
            continue
        if b.kind in ("para", "heading"):
            fb = {"document": dominant, "previous": prev}.get(o.neutral_direction, o.neutral_direction)
            d = detect_direction(b.text, fallback=fb) or fb
            prev = d
            if b.kind == "heading":
                mult = o.heading_scale[min(b.level, 6) - 1]
                body.append(_para_xml(b.text, d, o, style=f"Heading{min(b.level, 6)}", size_mult=mult))
            else:
                body.append(_para_xml(b.text, d, o))
        elif b.kind == "table":
            joined = " ".join(" ".join(r) for r in b.rows)
            td = detect_direction(joined, fallback=prev) or prev
            tbl = _table_xml(b.rows, td, o)
            if tbl:
                body.append(tbl)
                body.append("<w:p/>")   # Word requires a paragraph after a table
    if not body:
        body.append("<w:p/>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
          '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
          "</w:sectPr></w:body></w:document>"
    )


def export_docx(md: str, path: str, **options) -> None:
    """Write `md` to a Word file with correct per-paragraph RTL/LTR."""
    o = DocxOptions(**options)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", build_document_xml(md, o))
        z.writestr("word/styles.xml", _styles_xml(o))
        z.writestr("word/_rels/document.xml.rels", _DOC_RELS)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python rtl_export.py input.md output.docx")
        raise SystemExit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        export_docx(f.read(), sys.argv[2])
