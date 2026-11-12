#!/usr/bin/env python3
"""Search PubMed and GEO Series via NCBI EDirect and export a review CSV or HTML page."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template
from typing import Iterable


DEFAULT_EDIRECT = "/u/jhu30/edirect"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{}/"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}"
FIELDS = [
    "record_type", "short_name", "pmid", "geo_gse", "title", "authors", "year", "journal",
    "doi", "abstract_or_geo_summary", "content_summary", "matched_terms",
    "relevance_screen", "relevance_reason", "linked_pmids", "linked_gse",
    "pubmed_url", "geo_url",
]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in ("core_terms", "matched_term_patterns"):
        if not isinstance(cfg.get(key), list):
            raise ValueError(f"Keyword config must contain a {key} list")
    groups = cfg.get("context_groups")
    if not isinstance(groups, dict) or any(not isinstance(groups.get(k), list) for k in ("regulatory", "reporter")):
        raise ValueError("Keyword config must contain context_groups lists for regulatory and reporter")
    combos = cfg.get("context_combinations")
    if not isinstance(combos, dict) or any(not isinstance(combos.get(db), list) for db in ("pubmed", "gds")):
        raise ValueError("Keyword config must contain context_combinations lists for pubmed and gds")
    for combo in combos["pubmed"] + combos["gds"]:
        missing = [name for name in combo if not isinstance(groups.get(name), list)]
        if missing:
            raise ValueError(f"context_combinations refers to unknown context_groups: {', '.join(missing)}")
    return cfg


def quote_term(term: str) -> str:
    return '"' + term.replace('"', '\\"') + '"'


def grouped_terms(terms: list[str], field: str = "") -> str:
    suffix = f"[{field}]" if field else ""
    return "(" + " OR ".join(f"{quote_term(term)}{suffix}" for term in terms) + ")"


def context_query(cfg: dict, db: str, field: str = "") -> str:
    # Each combination ANDs its context groups; matching any one combination is enough.
    groups = cfg["context_groups"]
    return " OR ".join(
        "(" + " AND ".join(grouped_terms(groups[name], field) for name in combo) + ")"
        for combo in cfg["context_combinations"][db]
    )


def build_query(cfg: dict, db: str) -> str:
    context = cfg["context_groups"]
    regulatory = grouped_terms(context["regulatory"])
    reporters = grouped_terms(context["reporter"])
    if db == "gds":
        # GEO's GDS index does not use PubMed's Title/Abstract field tags.
        terms = [quote_term(x) for x in cfg["core_terms"]]
        terms.append(f'(MPRA AND {reporters} AND {regulatory})')
        return "(" + " OR ".join(terms + [context_query(cfg, "gds")]) + ") AND gse[ETYP]"
    # MPRA as a bare acronym is confined to title to reduce MprA gene collisions.
    terms = [f"{quote_term(x)}[Title/Abstract]" for x in cfg["core_terms"]]
    terms.append('MPRA[Title]')
    # Narrower catch-alls for studies that describe reporter or library assays
    # without the common platform names (see context_combinations in the config).
    # Platforms such as sort-seq are included by name, with the result screening score
    # providing a useful way to separate exact platform mentions from broad context.
    query = "(" + " OR ".join(terms + [context_query(cfg, "pubmed", "Title/Abstract")]) + ")"
    return query


def run_edirect(edirect: Path, args: list[str], input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            [str(edirect / args[0]), *args[1:]], input=input_text, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run EDirect command {args[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-8:]
        raise RuntimeError(
            f"EDirect {args[0]} failed (exit {result.returncode}): " + "\n".join(detail)
        )
    return result.stdout


def search_ids(edirect: Path, db: str, query: str, limit: int) -> list[str]:
    # nquire exposes ESearch's retmax parameter directly. This is important for
    # GDS: EDirect's usual streaming UID format is not supported by that database.
    response = run_edirect(edirect, [
        "nquire", "-url", EUTILS,
        "esearch.fcgi", "-db", db, "-term", query, "-retmax", str(limit), "-retmode", "xml",
    ])
    root = ET.fromstring(response)
    return list(dict.fromkeys((node.text or "").strip() for node in root.findall(".//IdList/Id") if (node.text or "").strip()))


def text_of(node: ET.Element | None, path: str, default: str = "") -> str:
    if node is None:
        return default
    found = node.find(path)
    if found is None:
        return default
    return " ".join("".join(found.itertext()).split())


def first_text(node: ET.Element, paths: Iterable[str]) -> str:
    for path in paths:
        value = text_of(node, path)
        if value:
            return value
    return ""


def joined_text(node: ET.Element | None, path: str, sep: str = "; ") -> str:
    if node is None:
        return ""
    vals = [" ".join("".join(e.itertext()).split()) for e in node.findall(path)]
    return sep.join(v for v in vals if v)


def short_name(surnames: list[str], year: str) -> str:
    """Citation-style name such as 'De Boer et al. 2020'."""
    if not surnames:
        return ""
    names = [s[:1].upper() + s[1:] for s in surnames[:2]]
    lead = names[0] if len(names) == 1 else " and ".join(names) if len(surnames) == 2 else names[0] + " et al."
    return f"{lead} {year}".strip()


def parse_pubmed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        article_node = citation.find("Article") if citation is not None else None
        if citation is None or article_node is None:
            continue
        pmid = text_of(citation, "PMID")
        title = text_of(article_node, "ArticleTitle")
        abstract = " ".join(
            " ".join("".join(item.itertext()).split())
            for item in article_node.findall("Abstract/AbstractText")
        )
        authors, surnames = [], []
        for author in article_node.findall("AuthorList/Author"):
            name = " ".join(filter(None, [text_of(author, "ForeName"), text_of(author, "LastName")]))
            if not name:
                name = text_of(author, "CollectiveName")
            if name:
                authors.append(name)
                surnames.append(text_of(author, "LastName") or name)
        journal = text_of(article_node, "Journal/Title")
        year = first_text(article_node, ["Journal/JournalIssue/PubDate/Year", "Journal/JournalIssue/PubDate/MedlineDate"])
        if not year:
            year = first_text(article_node, ["ArticleDate/Year"])
        dois = [x.text or "" for x in article.findall("PubmedData/ArticleIdList/ArticleId") if x.attrib.get("IdType") == "doi"]
        if not dois:
            dois = [x.text or "" for x in article_node.findall("ELocationID") if x.attrib.get("EIdType") == "doi"]
        gses = set(re.findall(r"\bGSE\d+\b", " ".join([title, abstract]), re.I))
        records.append({
            "record_type": "PubMed", "short_name": short_name(surnames, year[:4]),
            "pmid": pmid, "geo_gse": "; ".join(sorted({x.upper() for x in gses})),
            "title": title, "authors": "; ".join(authors), "year": year[:4], "journal": journal,
            "doi": "; ".join(dict.fromkeys(dois)), "abstract_or_geo_summary": abstract,
            "linked_pmids": pmid, "linked_gse": "; ".join(sorted({x.upper() for x in gses})),
        })
    return records


def parse_geo(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []
    docs = root.findall(".//DocumentSummary") or root.findall(".//DocSum")
    for doc in docs:
        values: dict[str, list[str]] = {}
        for item in list(doc):
            name = item.attrib.get("Name", "") or item.tag
            values.setdefault(name, []).append(" ".join("".join(item.itertext()).split()))
        def val(*names: str) -> str:
            return "; ".join(v for name in names for v in values.get(name, []) if v)
        uid = doc.get("uid") or text_of(doc, "Id")
        accession_values = re.findall(r"GSE\d+", val("Accession", "GSE"), re.I)
        accession = "; ".join(dict.fromkeys(item.upper() for item in accession_values))
        if not accession:
            accession = "GSE" + uid if uid and uid.isdigit() else ""
        summary = val("Summary", "summary", "Overall-Design", "Description")
        title = val("title", "Title")
        pubmed_ids = re.findall(r"\b\d{5,9}\b", val("PubMedIds", "PubmedIds", "PubMedID", "PubMed"))
        records.append({
            "record_type": "GEO Series", "gds_uid": uid, "short_name": "", "pmid": "; ".join(dict.fromkeys(pubmed_ids)),
            "geo_gse": accession, "title": title, "authors": val("Contact", "Contributor"),
            "year": val("PDAT", "ReleaseDate", "CreateDate")[:4], "journal": "", "doi": "",
            "abstract_or_geo_summary": summary,
            "linked_pmids": "; ".join(dict.fromkeys(pubmed_ids)), "linked_gse": accession,
        })
    return records


def elink_pubmed(edirect: Path, gds_uid: str) -> list[str]:
    try:
        xml_text = run_edirect(edirect, ["elink", "-db", "gds", "-target", "pubmed", "-id", gds_uid, "-cmd", "neighbor"])
        root = ET.fromstring(xml_text)
        return list(dict.fromkeys(
            node.text or "" for node in root.findall(".//LinkSetDb/Link/Id") if node.text
        ))
    except (RuntimeError, ET.ParseError):
        return []


def extractive_summary(text: str, max_chars: int = 520) -> str:
    if not text:
        return "No abstract/summary indexed in the retrieved record."
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
    selected = " ".join(sentences[:3]).strip()
    if len(selected) > max_chars:
        selected = selected[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return selected


def assess(record: dict, cfg: dict) -> tuple[str, str, str]:
    haystack = " ".join([record.get("title", ""), record.get("abstract_or_geo_summary", "")])
    matched = []
    for entry in cfg["matched_term_patterns"]:
        if re.search(entry["pattern"], haystack, flags=re.I):
            matched.append(entry["label"])
    distinctive = {entry["label"] for entry in cfg["matched_term_patterns"] if entry.get("distinctive")}
    is_review = bool(re.search(r"\b(review|perspective|protocol)\b", record.get("title", ""), re.I))
    context_terms = [term for terms in cfg["context_groups"].values() for term in terms]
    assay_terms = bool(re.search(r"\b(" + "|".join(map(re.escape, context_terms)) + r")\b", haystack, re.I))
    if distinctive.intersection(matched) and not is_review:
        rating, reason = "High", "Explicit MPRA-related platform term; confirm study type, assay design, and data availability."
    elif distinctive.intersection(matched) or (matched and assay_terms):
        rating, reason = "Review", "Related terminology found; inspect methods and supplementary data for a usable reporter dataset."
    else:
        rating, reason = "Review", "Broad reporter/regulatory-assay match; manual relevance check recommended."
    return "; ".join(matched), rating, reason


def chunked(items: list[str], size: int = 150) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_pubmed(edirect: Path, ids: list[str]) -> list[dict]:
    records = []
    for batch in chunked(ids):
        # Fetch raw E-utilities XML: EDirect's efetch reformatting can turn an
        # escaped "&amp;" in a reference URL into a bare "&", which is invalid XML.
        xml_text = run_edirect(edirect, [
            "nquire", "-url", EUTILS, "efetch.fcgi", "-db", "pubmed", "-id", ",".join(batch), "-retmode", "xml",
        ])
        records.extend(parse_pubmed(xml_text))
        time.sleep(0.35)
    return records


def fetch_geo(edirect: Path, ids: list[str]) -> list[dict]:
    records = []
    for batch in chunked(ids):
        # Raw E-utilities XML: EDirect's esummary splits large requests into several
        # concatenated XML documents and returns them sorted by UID.
        xml_text = run_edirect(edirect, [
            "nquire", "-url", EUTILS, "esummary.fcgi", "-db", "gds", "-id", ",".join(batch),
            "-version", "2.0", "-retmode", "xml",
        ])
        records.extend(parse_geo(xml_text))
        time.sleep(0.35)
    return records


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


HTML_PAGE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MPRA PubMed/GEO search</title>
<style>
:root { --bg: #fff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --head: #f6f8fa; --link: #0969da; --high: #1a7f37; --review: #9a6700; --chip: #eef1f4; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --head: #161b22; --link: #4493f8; --high: #3fb950; --review: #d29922; --chip: #262c36; }
}
body { margin: 0; padding: 20px 24px; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
.sub, .meta, .ids, #count { color: var(--muted); }
.meta, .ids { font-size: 12.5px; margin-top: 2px; }
pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; background: var(--head); padding: 8px; border-radius: 6px; }
.toolbar { position: sticky; top: 0; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 0; background: var(--bg); }
.toolbar input, .toolbar select { font: inherit; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--fg); }
.toolbar input { flex: 1 1 280px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 1080px; }
td.short { white-space: nowrap; font-weight: 600; }
th, td { text-align: left; vertical-align: top; padding: 8px; border-bottom: 1px solid var(--line); }
th { background: var(--head); cursor: pointer; user-select: none; white-space: nowrap; }
th[aria-sort="ascending"]::after { content: " \\25B2"; }
th[aria-sort="descending"]::after { content: " \\25BC"; }
td.title a.main { font-weight: 600; }
td.summary { max-width: 520px; }
.High { color: var(--high); font-weight: 600; }
.Review { color: var(--review); font-weight: 600; }
.chip { display: inline-block; background: var(--chip); border-radius: 10px; padding: 1px 8px; margin: 0 4px 4px 0; font-size: 12px; white-space: nowrap; }
details summary { cursor: pointer; color: var(--link); font-size: 12.5px; margin-top: 4px; }
.reason { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<h1>MPRA PubMed/GEO search</h1>
<div class="sub">Generated $generated &middot; $stats</div>
<details><summary>Search queries</summary>$queries</details>
<div class="toolbar">
  <input id="q" type="search" placeholder="Filter by title, abstract, authors, PMID or GSE">
  <select id="type"><option value="">All records</option><option>PubMed</option><option>GEO Series</option></select>
  <select id="screen"><option value="">Any screen</option><option>High</option><option>Review</option></select>
  <select id="term"><option value="">Any matched term</option>$term_options</select>
  <span id="count"></span>
</div>
<div class="wrap"><table>
<thead><tr><th>Screen</th><th>Type</th><th>Short name</th><th data-num>Year</th><th>Title</th><th>Matched terms</th><th>Summary</th></tr></thead>
<tbody>
$rows
</tbody>
</table></div>
<script>
const rows = Array.from(document.querySelectorAll("tbody tr"));
const body = document.querySelector("tbody");
const filters = ["q", "type", "screen", "term"].map(id => document.getElementById(id));
function apply() {
  const [q, type, screen, term] = filters.map(el => el.value.trim().toLowerCase());
  let shown = 0;
  for (const tr of rows) {
    const ok = (!q || tr.dataset.text.includes(q)) && (!type || tr.dataset.type === type)
      && (!screen || tr.dataset.screen === screen) && (!term || tr.dataset.terms.split("|").includes(term));
    tr.hidden = !ok;
    shown += ok;
  }
  document.getElementById("count").textContent = shown + " of " + rows.length + " records";
}
filters.forEach(el => el.addEventListener("input", apply));
document.querySelectorAll("th").forEach((th, col) => th.addEventListener("click", () => {
  const asc = th.getAttribute("aria-sort") !== "ascending";
  document.querySelectorAll("th").forEach(h => h.removeAttribute("aria-sort"));
  th.setAttribute("aria-sort", asc ? "ascending" : "descending");
  const key = tr => tr.cells[col].textContent.trim();
  const cmp = th.hasAttribute("data-num")
    ? (a, b) => (parseInt(key(a)) || 0) - (parseInt(key(b)) || 0)
    : (a, b) => key(a).localeCompare(key(b));
  rows.sort((a, b) => asc ? cmp(a, b) : cmp(b, a)).forEach(tr => body.appendChild(tr));
}));
apply();
</script>
</body>
</html>
""")


def split_ids(*values: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for value in values for x in value.split(";") if x.strip()))


def html_links(url: str, ids: list[str]) -> str:
    return ", ".join(f'<a href="{escape(url.format(x))}">{escape(x)}</a>' for x in ids)


def html_row(row: dict) -> str:
    pmids = [x for x in split_ids(row.get("pmid", ""), row.get("linked_pmids", "")) if x.isdigit()]
    gses = [x for x in split_ids(row.get("geo_gse", ""), row.get("linked_gse", "")) if x.startswith("GSE")]
    dois = split_ids(row.get("doi", ""))
    is_geo = row.get("record_type") == "GEO Series"
    url = GEO_URL.format(gses[0]) if is_geo and gses else PUBMED_URL.format(pmids[0]) if pmids else ""
    title = escape(row.get("title") or "(untitled)")
    authors = split_ids(row.get("authors", ""))
    byline = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    ids = [label + " " + links for label, links in (
        ("PMID", html_links(PUBMED_URL, pmids)), ("GEO", html_links(GEO_URL, gses)),
        ("DOI", html_links("https://doi.org/{}", dois)),
    ) if links]
    terms = split_ids(row.get("matched_terms", ""))
    abstract = row.get("abstract_or_geo_summary", "")
    text = " ".join([row.get("short_name", ""), row.get("title", ""), row.get("authors", ""), row.get("journal", ""), abstract, *pmids, *gses])
    return (
        f'<tr data-type="{escape(row.get("record_type", "").lower())}" data-screen="{escape(row.get("relevance_screen", "").lower())}"'
        f' data-terms="{escape("|".join(terms).lower())}" data-text="{escape(text.lower())}">'
        f'<td class="{escape(row.get("relevance_screen", ""))}">{escape(row.get("relevance_screen", ""))}</td>'
        f'<td>{escape(row.get("record_type", ""))}</td><td class="short">{escape(row.get("short_name", ""))}</td>'
        f'<td>{escape(row.get("year", ""))}</td>'
        f'<td class="title">' + (f'<a class="main" href="{escape(url)}">{title}</a>' if url else title)
        + f'<div class="meta">{escape(" · ".join(x for x in (byline, row.get("journal", "")) if x))}</div>'
        f'<div class="ids">{" · ".join(ids)}</div></td>'
        f'<td>{"".join(f"<span class=chip>{escape(t)}</span>" for t in terms)}</td>'
        f'<td class="summary">{escape(row.get("content_summary", ""))}'
        f'<details><summary>Full abstract and screening note</summary><p>{escape(abstract) or "&mdash;"}</p>'
        f'<p class="reason">{escape(row.get("relevance_reason", ""))}</p></details></td></tr>'
    )


def write_html(rows: list[dict], path: Path, queries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    types = [r.get("record_type") for r in rows]
    stats = (f"{len(rows)} records: {types.count('PubMed')} PubMed, {types.count('GEO Series')} GEO Series; "
             f"{sum(r.get('relevance_screen') == 'High' for r in rows)} rated High")
    terms = sorted({t for row in rows for t in split_ids(row.get("matched_terms", ""))}, key=str.lower)
    path.write_text(HTML_PAGE.substitute(
        generated=escape(datetime.now().strftime("%Y-%m-%d %H:%M")),
        stats=escape(stats),
        queries="".join(f"<p>{escape(db)}</p><pre>{escape(q)}</pre>" for db, q in queries.items()),
        term_options="".join(f'<option value="{escape(t.lower())}">{escape(t)}</option>' for t in terms),
        rows="\n".join(html_row(row) for row in rows),
    ), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edirect", default=DEFAULT_EDIRECT, help=f"EDirect directory (default: {DEFAULT_EDIRECT})")
    parser.add_argument("--keywords", type=Path, default=Path(__file__).with_name("mpra_keywords.json"), help="JSON keyword config")
    parser.add_argument("--output", type=Path, help="Output path (default: mpra_pubmed_geo.csv or .html next to this script)")
    parser.add_argument("--format", choices=("csv", "html"), default="csv",
                        help="csv table, or a readable html page with filters and links (default: csv)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum records from each database (default: 50)")
    parser.add_argument("--sample", action="store_true", help="Small validation run; same as --limit 3")
    parser.add_argument("--pubmed-only", action="store_true")
    parser.add_argument("--geo-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.pubmed_only and args.geo_only:
        parser.error("Choose at most one of --pubmed-only and --geo-only")
    edirect = Path(args.edirect).expanduser()
    for binary in ("esearch", "efetch", "esummary", "elink", "nquire"):
        if not (edirect / binary).exists():
            parser.error(f"EDirect executable not found: {edirect / binary}")
    output = args.output or Path(__file__).with_name(f"mpra_pubmed_geo.{args.format}")
    cfg = load_config(args.keywords)
    limit = 3 if args.sample else args.limit
    all_rows: list[dict] = []
    queries: dict[str, str] = {}
    if not args.geo_only:
        q = queries["PubMed"] = build_query(cfg, "pubmed")
        print(f"Searching PubMed (maximum {limit})…", file=sys.stderr)
        pubmed_ids = search_ids(edirect, "pubmed", q, limit)
        pubmed = fetch_pubmed(edirect, pubmed_ids)
        all_rows.extend(pubmed)
        print(f"Retrieved {len(pubmed)} PubMed records.", file=sys.stderr)
    else:
        pubmed = []
    if not args.pubmed_only:
        q = queries["GEO"] = build_query(cfg, "gds")
        print(f"Searching GEO Series (maximum {limit})…", file=sys.stderr)
        geo_ids = search_ids(edirect, "gds", q, limit)
        geo = fetch_geo(edirect, geo_ids)
        for row in geo:
            linked = elink_pubmed(edirect, row["gds_uid"])
            row["linked_pmids"] = "; ".join(linked) or row.get("linked_pmids", "")
            row["pmid"] = row["linked_pmids"]
            for pmid in linked:
                for article in pubmed:
                    if article.get("pmid") == pmid:
                        article["linked_gse"] = "; ".join(filter(None, [article.get("linked_gse", ""), row.get("geo_gse", "")]))
            time.sleep(0.34)
        # GEO summaries carry no authors, so name each series after its linked papers,
        # fetching any that are not already among the PubMed results.
        names = {article["pmid"]: article["short_name"] for article in pubmed}
        missing = sorted({pmid for row in geo for pmid in split_ids(row["linked_pmids"]) if pmid not in names})
        names.update((article["pmid"], article["short_name"]) for article in fetch_pubmed(edirect, missing))
        for row in geo:
            row["short_name"] = "; ".join(dict.fromkeys(
                names[pmid] for pmid in split_ids(row["linked_pmids"]) if names.get(pmid)
            ))
        all_rows.extend(geo)
        print(f"Retrieved {len(geo)} GEO Series records.", file=sys.stderr)
    # Match GSE accessions directly mentioned in publication abstracts/titles to GEO results.
    known_gses = {r.get("geo_gse", "").upper() for r in all_rows if r.get("record_type") == "GEO Series"}
    for row in all_rows:
        mentioned = set(re.findall(r"\bGSE\d+\b", " ".join([row.get("title", ""), row.get("abstract_or_geo_summary", "")]), re.I))
        if row.get("record_type") == "PubMed":
            matches = sorted(mentioned.intersection(known_gses))
            if matches:
                row["linked_gse"] = "; ".join(sorted(set(filter(None, [row.get("linked_gse", ""), *matches]))))
    # Deduplicate by stable accession and preserve one row per database record.
    deduped: dict[tuple[str, str], dict] = {}
    for row in all_rows:
        key = (row.get("record_type", ""), row.get("pmid", "") if row.get("record_type") == "PubMed" else row.get("geo_gse", ""))
        if key[1]:
            deduped[key] = row
    final_rows = list(deduped.values())
    for row in final_rows:
        row["content_summary"] = extractive_summary(row.get("abstract_or_geo_summary", ""))
        matched, rating, reason = assess(row, cfg)
        row["matched_terms"] = matched
        row["relevance_screen"] = rating
        row["relevance_reason"] = reason
        row["pubmed_url"] = "; ".join(PUBMED_URL.format(x) for x in row.get("pmid", "").split("; ") if x.isdigit())
        row["geo_url"] = "; ".join(GEO_URL.format(x) for x in row.get("geo_gse", "").split("; ") if x.startswith("GSE"))
    final_rows.sort(key=lambda r: (r.get("relevance_screen") != "High", r.get("record_type", ""), r.get("year", ""), r.get("title", "")), reverse=False)
    if args.format == "html":
        write_html(final_rows, output, queries)
    else:
        write_csv(final_rows, output)
    print(f"Wrote {len(final_rows)} deduplicated rows to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, ET.ParseError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
