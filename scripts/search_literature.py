#!/usr/bin/env python
r"""Search PubMed and GEO for keywords and build tables for curating MPRA papers.

    python scripts/search_literature.py \
        -k MPRA "massively parallel reporter assay" STARR-seq sort-seq lentiMPRA \
        -o /projects/bjgn/jhu30/lit_search

Writes to OUTDIR:
    pubmed.csv    one row per paper (keyword hits + papers linked from GEO hits)
    geo.csv       one row per GEO series (GSE); `paired_pmids` lists its papers
                  in pubmed.csv
    report.html   both tables, sortable and filterable, for browsing; the GEO
                  tab leaves out series already listed under a paper
    queries.txt   the exact Entrez queries that were run
    raw/          the downloaded XML

Papers and series are cross-linked: each paper lists its GEO series and each
series lists its papers. Rows are sorted by `score`, a rough relevance heuristic
(keyword hits in the title count most, linked GEO data adds, reviews subtract,
and `off_topic` records that never mention regulatory genomics drop to the end).
`key_sentences` is an extractive summary: the abstract sentences that best say
what was done ("Here we ...", conclusions, sentences with keywords or numbers).

Needs NCBI EDirect (esearch, efetch, esummary, nquire) on PATH. Set NCBI_API_KEY
to raise the NCBI request limit from 3/s to 10/s.
"""
import argparse
import csv
import html
import re
import string
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BATCH = 200
# Papers linked to more GEO series than this are consortium/reference papers
# (ENCODE 2012 has >20,000), not the paper that describes a series.
HUB_LINKS = 50

# Informational columns. Patterns are matched case-insensitively on whole words.
ASSAYS = {
    "MPRA": r"MPRAs?|massively parallel reporter assays?",
    "lentiMPRA": r"lenti-?MPRAs?",
    "scMPRA": r"single[- ]cell MPRAs?|scMPRAs?",
    "STARR-seq": r"STARR-?seq",
    "Sort-seq": r"sort-?seq",
    "SuRE": r"(?-i:SuRE)|survey of regulatory elements",
    "CRE-seq": r"CRE-?seq",
    "FIREWACh": r"FIREWACh",
    "saturation mutagenesis": r"saturation mutagenesis",
    "CRISPR screen": r"CRISPR(?:i|a)? screens?",
}
CELL_TYPES = [
    "K562", "HepG2", "HEK293T?", "GM12878", "HeLa", "A549", "MCF-?7", "SK-N-SH",
    "HCT116", "WTC11", "Jurkat", "THP-1", "LNCaP", "22Rv1", "IMR-90", "HUVECs?",
    "Caco-2", "Neuro-?2a", "C2C12", "NIH ?3T3", "S2 cells?", "iPSCs?", "hESCs?",
    "mESCs?", "ESCs?", "NPCs?", "neurons?", "cardiomyocytes?", "hepatocytes?",
    "keratinocytes?", "macrophages?", "T cells?", "B cells?", "chondrocytes?",
    "organoids?", "zebrafish embryos?", "mouse retina", "mouse liver", "mouse brain",
]
ORGANISMS = {
    "human": r"humans?|Homo sapiens",
    "mouse": r"mouse|mice|murine|Mus musculus",
    "rat": r"rats?|Rattus",
    "fly": r"Drosophila",
    "zebrafish": r"zebrafish|Danio rerio",
    "worm": r"C\. elegans|Caenorhabditis",
    "yeast": r"yeast|Saccharomyces|S\. cerevisiae",
    "bacteria": r"E\. coli|Escherichia coli|bacteria\w*",
    "plant": r"Arabidopsis|maize|Zea mays|tobacco|plants?",
}
SCALE_UNITS = (
    r"variants?|elements?|enhancers?|promoters?|sequences?|oligos?|"
    r"oligonucleotides?|fragments?|SNPs?|SNVs?|CREs?|regions?|alleles?|mutations?|"
    r"mutants?|constructs?|barcodes?|haplotypes?|tiles?|motifs?|indels?|designs?"
)
SCALE_RE = re.compile(
    r"(?:~|>|over |more than |nearly |approximately |about |up to )?"
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*(?:million|thousand|[kKM])\b)?"
    r"\s+(?:[A-Za-z][\w-]*\s+){0,2}?(?:" + SCALE_UNITS + r")\b"
)
HERE_WE_RE = re.compile(
    r"\bhere,? we\b|\bin this (?:study|work|paper)\b|\bwe (?:present|report|describe|"
    r"develop\w*|perform\w*|appl\w*|use[d]?|test\w*|measur\w*|screen\w*|"
    r"characteriz\w*|introduc\w*|identif\w*|generat\w*|design\w*|assay\w*)\b",
    re.I,
)
FINDING_RE = re.compile(
    r"\bwe (?:find|found|show\w*|reveal\w*|demonstrat\w*|observ\w*)\b|"
    r"\bour (?:results|data|findings)\b",
    re.I,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# Off-topic uses of the keywords (e.g. MPRA = "modified periodontal risk
# assessment", the MprA protease) rarely mention any of these.
CONTEXT_RE = re.compile(
    r"enhancer|promoter|regulat|transcription|reporter|barcod|variant|cis-|"
    r"expression|chromatin|silencer|motif|sequence-function|genom|massively parallel",
    re.I,
)


def words_re(pattern):
    return re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I)


ASSAY_RES = {name: words_re(p) for name, p in ASSAYS.items()}
CELL_RES = {re.sub(r".\?", "", c): words_re(c) for c in CELL_TYPES}  # "HEK293T?" -> "HEK293"
ORGANISM_RES = {name: words_re(p) for name, p in ORGANISMS.items()}


# --------------------------------------------------------------------- EDirect

def run(cmd, stdin=None):
    """Run an EDirect command and return its stdout."""
    # EDirect reads stdin whenever it is not a terminal, so never inherit ours.
    res = subprocess.run(cmd, input=stdin, stdin=None if stdin else subprocess.DEVNULL,
                         capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd[:4])} ...\n{res.stderr[-2000:]}")
    return res.stdout


def eutils(endpoint, **params):
    """Call an E-utility through `nquire`; list values become repeated params."""
    cmd = ["nquire", "-url", EUTILS, endpoint]
    for key, value in params.items():
        for v in value if isinstance(value, list) else [value]:
            cmd += [f"-{key}", str(v)]
    out = run(cmd)
    time.sleep(0.35)
    return out


def batches(items, size=BATCH):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_pubmed_by_query(query):
    hits = run(["esearch", "-db", "pubmed", "-query", query])
    return run(["efetch", "-format", "xml"], stdin=hits)


def fetch_pubmed_by_ids(pmids):
    return [run(["efetch", "-db", "pubmed", "-id", ",".join(b), "-format", "xml"])
            for b in batches(pmids)]


def search_geo(query, retmax):
    # `esearch -db gds | esummary` fails in EDirect 26.5: for gds, `efetch -format
    # uid` returns text summaries instead of UIDs. Get the UIDs from esearch.fcgi.
    root = parse_xml(eutils("esearch.fcgi", db="gds", term=query, retmax=retmax))
    count = int(root.findtext("eSearchResult/Count", "0"))
    if count > retmax:
        print(f"  WARNING: {count} GEO hits, keeping the first {retmax} "
              "(raise --max-geo)", file=sys.stderr)
    return [e.text for e in root.findall("eSearchResult/IdList/Id")]


def fetch_geo_docsums(uids):
    return [run(["esummary", "-db", "gds", "-id", ",".join(b)]) for b in batches(uids)]


def link_pubmed_to_geo(pmids):
    """Map each PMID to the GEO series (GSE) linked to it in Entrez."""
    links = {}
    for b in batches(pmids, 100):
        root = parse_xml(eutils("elink.fcgi", dbfrom="pubmed", db="gds", id=b))
        for ls in root.iter("LinkSet"):
            pmid = ls.findtext("IdList/Id")
            # gds UIDs 2xxxxxxxx are series: 200012345 -> GSE12345.
            uids = [int(e.text) for e in ls.iterfind("LinkSetDb/Link/Id")]
            links[pmid] = {f"GSE{u - 200000000}" for u in uids
                           if 200000000 <= u < 300000000}
    return links


# --------------------------------------------------------------------- parsing

# NCBI records occasionally contain bare '&' (e.g. URLs in reference lists).
BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")
PROLOG_RE = re.compile(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>")


def parse_xml(xml_text):
    # EDirect splits large requests and concatenates the XML documents it gets
    # back, so drop the prologs and wrap everything in one root.
    xml_text = BARE_AMP_RE.sub("&amp;", PROLOG_RE.sub("", xml_text))
    return ET.fromstring(f"<root>{xml_text}</root>")


def text(elem):
    return " ".join("".join(elem.itertext()).split()) if elem is not None else ""


def parse_pubmed(xml_text):
    papers = {}
    for art in parse_xml(xml_text).iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        a = cit.find("Article")
        pmid = cit.findtext("PMID")

        abstract = []
        for t in a.iterfind("Abstract/AbstractText"):
            label = t.get("Label")
            abstract.append((label.upper() if label else "", text(t)))

        authors = []
        for au in a.iterfind("AuthorList/Author"):
            if au.find("CollectiveName") is not None:
                authors.append(text(au.find("CollectiveName")))
            elif au.findtext("LastName"):
                authors.append(f"{au.findtext('LastName')} {au.findtext('Initials', '')}".strip())

        pub = a.find("Journal/JournalIssue/PubDate")
        year = (pub.findtext("Year") or pub.findtext("MedlineDate", "")[:4]
                or a.findtext("ArticleDate/Year", ""))

        ids = {i.get("IdType"): i.text for i in art.iterfind("PubmedData/ArticleIdList/ArticleId")}
        geo = {acc.text for db in a.iterfind("DataBankList/DataBank")
               if db.findtext("DataBankName") == "GEO"
               for acc in db.iterfind("AccessionNumberList/AccessionNumber")
               if acc.text and acc.text.startswith("GSE")}

        papers[pmid] = {
            "pmid": pmid,
            "title": text(a.find("ArticleTitle")),
            "abstract_parts": abstract,
            "authors": authors,
            "journal": a.findtext("Journal/ISOAbbreviation") or a.findtext("Journal/Title", ""),
            "year": year,
            "doi": ids.get("doi", ""),
            "pmcid": ids.get("pmc", ""),
            "pub_types": [text(p) for p in a.iterfind("PublicationTypeList/PublicationType")],
            "mesh": [text(m) for m in cit.iterfind("MeshHeadingList/MeshHeading/DescriptorName")],
            "keywords": [text(k) for k in cit.iterfind("KeywordList/Keyword")],
            "geo": geo,
        }
    return papers


def parse_geo(xml_text):
    series = {}
    for ds in parse_xml(xml_text).iter("DocumentSummary"):
        if ds.findtext("entryType") != "GSE":
            continue
        acc = ds.findtext("Accession")
        series[acc] = {
            "accession": acc,
            "title": text(ds.find("title")),
            "summary": text(ds.find("summary")),
            "organism": ds.findtext("taxon", ""),
            "experiment_type": ds.findtext("gdsType", ""),
            "year": ds.findtext("PDAT", "")[:4],
            "n_samples": ds.findtext("n_samples", ""),
            "supp_files": ds.findtext("suppFile", ""),
            "bioproject": ds.findtext("BioProject", ""),
            "pmids": [e.text for e in ds.iterfind("PubMedIds/int")],
        }
    return series


# --------------------------------------------------------------- summarization

def keyword_variants(kw):
    """'sort-seq' -> sort-seq, sortseq, sort seq, plus plurals."""
    forms = {kw, kw.replace("-", ""), kw.replace("-", " ")}
    return sorted(forms | {f + "s" for f in forms})


def keyword_re(kw):
    parts = [re.escape(p) for p in re.split(r"[-\s]+", kw)]
    return words_re(r"[-\s]?".join(parts) + "s?")


def find_all(regexes, txt):
    return [name for name, rx in regexes.items() if rx.search(txt)]


def scale_phrases(txt, limit=3):
    """Library sizes like '25,892 CAD-associated variants' (numbers >= 50)."""
    out = {}
    for m in SCALE_RE.finditer(txt):
        num = m.group(1).replace(",", "")
        if (m.group(2) or float(num) >= 50) and num not in out:
            out[num] = m.group(0).strip()
    return list(out.values())[:limit]


def key_sentences(parts, kw_res, n=2, max_len=350):
    """Pick the n sentences that best describe what the study did."""
    sents = [(label, s) for label, t in parts for s in SENTENCE_RE.split(t) if s]
    scored = []
    for i, (label, s) in enumerate(sents):
        score = 0
        score += 3 if HERE_WE_RE.search(s) else 0
        score += 1 if FINDING_RE.search(s) else 0
        score += 2 if any(rx.search(s) for rx in kw_res.values()) else 0
        score += 1 if SCALE_RE.search(s) else 0
        score += 2 if label.startswith("CONCLUSION") else 0
        score += 1 if label.startswith("RESULT") else 0
        score -= 2 if label.startswith(("BACKGROUND", "INTRODUCTION", "CONTEXT")) else 0
        # Unstructured abstracts usually open with background.
        score -= 1 if not label and i < len(sents) / 4 else 0
        scored.append((score, i))
    best = sorted(i for _, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:n])
    return " … ".join(s if len(s) <= max_len else s[:max_len] + "…"
                      for s in (sents[i][1] for i in best))


def annotate(title, body, kw_res):
    """Columns shared by papers and GEO series."""
    full = f"{title} {body}"
    in_title = find_all(kw_res, title)
    in_body = find_all(kw_res, body)
    mentions = sum(len(rx.findall(full)) for rx in kw_res.values())
    on_topic = bool(CONTEXT_RE.search(full))
    return {
        "matched_keywords": "; ".join(k for k in kw_res if k in in_title or k in in_body),
        "assays": "; ".join(find_all(ASSAY_RES, full)),
        "cell_types": "; ".join(find_all(CELL_RES, full)),
        "scale": "; ".join(scale_phrases(full)),
        "off_topic": "" if on_topic else "likely",
        "score": 3 * len(in_title) + 2 * len(in_body) + min(mentions, 5) - (0 if on_topic else 5),
    }


def cite(p):
    first = p["authors"][0] if p["authors"] else "?"
    etal = " et al." if len(p["authors"]) > 1 else ""
    return f"{first}{etal} {p['year']} {p['journal']}: {p['title']}"


def cap(items, n=10):
    """Some papers (e.g. ENCODE) link to thousands of GEO series."""
    more = f"; ... (+{len(items) - n} more)" if len(items) > n else ""
    return "; ".join(items[:n]) + more


def short_authors(authors, n=3):
    if len(authors) <= n + 1:
        return ", ".join(authors)
    return ", ".join(authors[:n]) + ", ..., " + authors[-1]


def paper_row(p, kw_res, geo_hits):
    abstract = " ".join(f"{label}: {t}" if label else t for label, t in p["abstract_parts"])
    # PubMed's [tiab] also searches author keywords, so match against them too.
    row = annotate(p["title"], f"{abstract} {'; '.join(p['keywords'])}", kw_res)
    is_review = any(t in ("Review", "Systematic Review") for t in p["pub_types"])
    row["score"] += (2 if p["geo"] else 0) - (3 if is_review else 0) - (2 if not abstract else 0)
    organisms = find_all(ORGANISM_RES, f"{p['title']} {abstract} {' '.join(p['mesh'])}")
    geo = sorted(p["geo"] & geo_hits) + sorted(p["geo"] - geo_hits)
    return {
        "score": row["score"],
        "pmid": p["pmid"],
        "year": p["year"],
        "title": p["title"],
        "first_author": p["authors"][0] if p["authors"] else "",
        "last_author": p["authors"][-1] if p["authors"] else "",
        "authors": short_authors(p["authors"]),
        "journal": p["journal"],
        "pub_type": "; ".join(t for t in p["pub_types"] if t != "Journal Article"
                              and not t.startswith("Research Support")),
        "is_review": "yes" if is_review else "",
        "source": p["source"],
        "off_topic": row["off_topic"],
        "matched_keywords": row["matched_keywords"],
        "assays": row["assays"],
        "organisms": "; ".join(organisms),
        "cell_types": row["cell_types"],
        "scale": row["scale"],
        "n_geo_series": len(p["geo"]),
        "geo_series_in_search": "; ".join(sorted(p["geo"] & geo_hits)),
        "geo_series": "; ".join(geo) if len(geo) <= HUB_LINKS else cap(geo),
        "key_sentences": key_sentences(p["abstract_parts"], kw_res),
        "abstract": abstract,
        "author_keywords": "; ".join(p["keywords"]),
        "doi": p["doi"],
        "pmcid": p["pmcid"],
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/",
        "relevant": "",
        "notes": "",
    }


def geo_row(g, kw_res, papers, paired_pmids):
    row = annotate(g["title"], g["summary"], kw_res)
    row["score"] += 1 if g["pmids"] else 0
    return {
        "score": row["score"],
        "accession": g["accession"],
        "year": g["year"],
        "title": g["title"],
        "organism": g["organism"],
        "experiment_type": g["experiment_type"],
        "n_samples": g["n_samples"],
        "off_topic": row["off_topic"],
        "matched_keywords": row["matched_keywords"],
        "assays": row["assays"],
        "cell_types": row["cell_types"],
        "scale": row["scale"],
        "key_sentences": key_sentences([("", g["summary"])], kw_res),
        "pmids": "; ".join(g["pmids"]),
        "papers": " | ".join(cite(papers[i]) for i in g["pmids"] if i in papers),
        "paired_pmids": "; ".join(sorted(paired_pmids)),
        "summary": g["summary"],
        "supp_files": g["supp_files"],
        "bioproject": g["bioproject"],
        "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g['accession']}",
        "relevant": "",
        "notes": "",
    }


# ---------------------------------------------------------------------- output

def save_raw(outdir, name, xml_batches):
    for i, x in enumerate(xml_batches):
        (outdir / "raw" / f"{name}_{i:03d}.xml").write_text(x)


def write_csv(rows, path):
    if not rows:
        path.write_text("")
        return
    # utf-8-sig so Excel detects the encoding.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# Columns shown in the HTML report; long text goes into an expandable cell.
HTML_COLS = {
    "pubmed": ["score", "pmid", "year", "title", "authors", "journal", "pub_type",
               "source", "off_topic", "matched_keywords", "assays", "organisms", "cell_types",
               "scale", "geo_series", "key_sentences", "abstract"],
    "geo": ["score", "accession", "year", "title", "organism", "experiment_type",
            "n_samples", "matched_keywords", "assays", "cell_types", "scale",
            "pmids", "papers", "key_sentences", "summary"],
}
LONG_COLS = {"abstract", "summary", "papers"}
LINKS = {
    "pmid": "https://pubmed.ncbi.nlm.nih.gov/{}/",
    "pmids": "https://pubmed.ncbi.nlm.nih.gov/{}/",
    "accession": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}",
    "geo_series": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}",
}


def html_cell(col, value):
    value = str(value)
    if col in LINKS and value:
        return "; ".join(f'<a href="{LINKS[col].format(v)}" target="_blank">{v}</a>'
                         if re.fullmatch(r"\w+", v) else html.escape(v)
                         for v in value.split("; "))
    if col in LONG_COLS and value:
        return f"<details><summary>show</summary>{html.escape(value)}</details>"
    return html.escape(value)


def html_table(name, rows):
    cols = HTML_COLS[name]
    head = "".join(f"<th data-col='{i}'>{c}</th>" for i, c in enumerate(cols))
    body = "\n".join(
        "<tr>" + "".join(f"<td class='{c}'>{html_cell(c, r[c])}</td>" for c in cols) + "</tr>"
        for r in rows)
    return (f"<table id='{name}' class='tab'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>")


HTML_TEMPLATE = string.Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>Literature search</title>
<style>
body {font: 13px system-ui, sans-serif; margin: 16px; color: #222; background: #fff;}
.bar {position: sticky; top: 0; background: #fff; padding: 8px 0; z-index: 2;}
.bar button.on {font-weight: bold;}
input {padding: 4px; width: 320px;}
table {border-collapse: collapse; width: 100%; display: none;}
table.on {display: table;}
th, td {border: 1px solid #ddd; padding: 4px 6px; vertical-align: top; text-align: left;}
th {background: #f3f3f3; cursor: pointer; position: sticky; top: 44px;}
td.title {min-width: 240px; font-weight: 600;}
td.key_sentences {min-width: 360px;}
td.abstract, td.summary {min-width: 120px; max-width: 600px;}
tr:hover td {background: #fafaf0;}
</style></head><body>
<h2>Literature search</h2>
<p>Keywords: $keywords<br>Generated $date. Click a column header to sort.<br>
The GEO tab leaves out $n_paired series that are already listed under a paper in
the PubMed tab (geo_series column); geo.csv has all of them.</p>
<div class="bar">
  <button data-t="pubmed" class="on">PubMed papers ($n_pubmed)</button>
  <button data-t="geo">GEO series without a paper ($n_geo)</button>
  <input id="q" placeholder="filter rows (text, e.g. K562 or lentiMPRA)">
  min score <input id="min" type="number" value="0" style="width:60px">
  <span id="n"></span>
</div>
$tables
<script>
let cur = 'pubmed';
const q = document.getElementById('q'), min = document.getElementById('min');
function filter() {
  const t = document.getElementById(cur), s = q.value.toLowerCase(), m = +min.value || 0;
  let n = 0;
  for (const tr of t.tBodies[0].rows) {
    const ok = (+tr.cells[0].textContent >= m) && (!s || tr.textContent.toLowerCase().includes(s));
    tr.style.display = ok ? '' : 'none'; n += ok;
  }
  document.getElementById('n').textContent = n + ' rows';
}
document.querySelectorAll('.bar button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.bar button, table').forEach(e => e.classList.remove('on'));
  b.classList.add('on'); cur = b.dataset.t;
  document.getElementById(cur).classList.add('on'); filter();
});
document.querySelectorAll('th').forEach(th => th.onclick = () => {
  const tb = th.closest('table').tBodies[0], i = +th.dataset.col;
  const dir = th.dataset.dir = th.dataset.dir === 'desc' ? 'asc' : 'desc';
  const val = tr => { const v = tr.cells[i].textContent.trim(); return v !== '' && !isNaN(v) ? +v : v.toLowerCase(); };
  [...tb.rows].sort((a, b) => { const x = val(a), y = val(b); return (x > y) - (x < y); })
    .forEach(tr => dir === 'desc' ? tb.prepend(tr) : tb.append(tr));
});
q.oninput = min.oninput = filter;
document.getElementById('pubmed').classList.add('on'); filter();
</script></body></html>
""")


def write_html(paper_rows, geo_rows, keywords, path):
    unpaired = [r for r in geo_rows if not r["paired_pmids"]]
    page = HTML_TEMPLATE.substitute(
        keywords=html.escape(", ".join(keywords)),
        date=time.strftime("%Y-%m-%d"),
        n_pubmed=len(paper_rows),
        n_geo=len(unpaired),
        n_paired=len(geo_rows) - len(unpaired),
        tables=html_table("pubmed", paper_rows) + html_table("geo", unpaired),
    )
    path.write_text(page, encoding="utf-8")


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--keywords", nargs="+", required=True,
                    help="Keywords or phrases; a record matches if it contains any of them.")
    ap.add_argument("-o", "--outdir", default="lit_search", help="Output folder.")
    ap.add_argument("--mindate", default="1800", help="Earliest publication date, YYYY[/MM[/DD]].")
    ap.add_argument("--maxdate", default="3000", help="Latest publication date, YYYY[/MM[/DD]].")
    ap.add_argument("--max-geo", type=int, default=5000, help="Max GEO series to fetch.")
    ap.add_argument("--no-geo", action="store_true", help="Search PubMed only.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    kw_res = {kw: keyword_re(kw) for kw in args.keywords}
    variants = [v for kw in args.keywords for v in keyword_variants(kw)]
    dates = f'("{args.mindate}"[PDAT] : "{args.maxdate}"[PDAT])'
    pubmed_query = "(" + " OR ".join(f'"{v}"[tiab]' for v in variants) + f") AND {dates}"
    geo_query = "(" + " OR ".join(f'"{v}"' for v in variants) + f') AND gse[ETYP] AND {dates}'
    (outdir / "queries.txt").write_text(f"PubMed:\n{pubmed_query}\n\nGEO:\n{geo_query}\n")

    print("Searching PubMed ...", file=sys.stderr)
    xml_text = fetch_pubmed_by_query(pubmed_query)
    (outdir / "raw" / "pubmed.xml").write_text(xml_text)
    papers = parse_pubmed(xml_text)
    for p in papers.values():
        p["source"] = "keyword"
    print(f"  {len(papers)} papers", file=sys.stderr)

    series = {}
    if not args.no_geo:
        print("Searching GEO ...", file=sys.stderr)
        uids = search_geo(geo_query, args.max_geo)
        docsums = fetch_geo_docsums(uids)
        save_raw(outdir, "geo_docsum", docsums)
        for d in docsums:
            series.update(parse_geo(d))
        print(f"  {len(series)} series", file=sys.stderr)

        # Papers behind GEO hits often describe the assay without the keywords.
        extra = sorted({i for g in series.values() for i in g["pmids"]} - set(papers))
        if extra:
            print(f"Fetching {len(extra)} papers linked from GEO series ...", file=sys.stderr)
            extra_xml = fetch_pubmed_by_ids(extra)
            save_raw(outdir, "pubmed_from_geo", extra_xml)
            for x in extra_xml:
                for pmid, p in parse_pubmed(x).items():
                    p["source"] = "GEO link"
                    papers[pmid] = p

    print("Linking papers to GEO series ...", file=sys.stderr)
    for pmid, gses in link_pubmed_to_geo(list(papers)).items():
        if pmid in papers:
            papers[pmid]["geo"] |= gses
    for g in series.values():
        for pmid in g["pmids"]:
            if pmid in papers:
                papers[pmid]["geo"].add(g["accession"])

    paired = {}
    for p in papers.values():
        if len(p["geo"]) <= HUB_LINKS:
            for acc in p["geo"]:
                paired.setdefault(acc, []).append(p["pmid"])

    def order(r):
        return (-r["score"], -int(r["year"]) if r["year"].isdigit() else 0)

    paper_rows = sorted((paper_row(p, kw_res, set(series)) for p in papers.values()), key=order)
    geo_rows = sorted((geo_row(g, kw_res, papers, paired.get(acc, []))
                       for acc, g in series.items()), key=order)
    write_csv(paper_rows, outdir / "pubmed.csv")
    write_csv(geo_rows, outdir / "geo.csv")
    write_html(paper_rows, geo_rows, args.keywords, outdir / "report.html")
    print(f"Wrote {len(paper_rows)} papers and {len(geo_rows)} GEO series to {outdir}/",
          file=sys.stderr)


if __name__ == "__main__":
    main()
