# PubMed and GEO MPRA search

`mpra_geo_search.py` uses the installed NCBI EDirect commands to search PubMed and GEO Series, retrieve article and series metadata, and write either a UTF-8 CSV that opens cleanly in Excel or a readable HTML page. The default EDirect location is `/u/jhu30/edirect`.

## First small validation run

From this folder, run:

```bash
python3 mpra_geo_search.py --sample --output mpra_sample.csv
```

This fetches up to three records from each database. After reviewing the behavior, run a larger search:

```bash
python3 mpra_geo_search.py --limit 200 --output mpra_pubmed_geo.csv
```

The default limit is 50 records per database. Use `--pubmed-only` or `--geo-only` to search one database. To use a different EDirect directory, add `--edirect /path/to/edirect`.

## HTML output

Add `--format html` to write a single self-contained HTML page instead of the CSV:

```bash
python3 mpra_geo_search.py --limit 200 --format html --output mpra_pubmed_geo.html
```

Without `--output`, the file is written next to the script as `mpra_pubmed_geo.html`. The page has one row per record with linked PMIDs, GSE accessions and DOIs, matched-term tags, the short summary, and an expandable full abstract/GEO summary with the screening note. A search box and filters for record type, relevance screen and matched term narrow the table, and clicking a column header sorts it. The exact queries that were run are listed at the top.

## Search terms and precision

Edit `mpra_keywords.json` to change the search. It has four parts:

- `core_terms`: named methods searched on their own (MPRA/massively parallel reporter, STARR-seq, lentiMPRA, scMPRA, sort-seq, CRE-seq, random promoters, massively parallel splicing, Vex-seq, fast-UTR, PTRE-seq). The bare acronym MPRA is constrained by reporter/regulatory context in GEO and searched in PubMed titles to reduce unrelated MprA gene matches.
- `context_groups`: named term lists. `regulatory` holds DNA elements (enhancers, promoters, silencers, insulators), `rna_regulatory` holds RNA-level regulation (5'/3' UTRs, splicing, polyadenylation, translation efficiency/ribosome load, mRNA stability), and `reporter`, `throughput`, `library` and `parallel_readout` describe how a high-throughput assay is worded. Quoted terms are not stemmed, so list plural forms explicitly. `UTR` also matches `5' UTR`, `5'UTR` and `3' UTR`.
- `context_combinations`: for each database, a list of group combinations. The groups within one combination are ANDed and the combinations are ORed, so for example `["rna_regulatory", "library", "parallel_readout"]` finds a splicing or UTR study that describes a sequence library with a massively parallel or polysome-profiling readout. GEO uses fewer combinations than PubMed because GEO summaries use "library" for sequencing libraries.
- `matched_term_patterns`: labels and regular expressions for the `matched_terms` column. Entries marked `"distinctive": true` are named platforms that earn a `High` relevance screen; the others tag the regulatory type (5' UTR, 3' UTR, splicing, polyadenylation, translation / ribosome load, mRNA stability) or broader assay wording.

The RNA-level terms were checked against the papers behind the datasets in `/projects/bjgn/jhu30/MPRACollection_data/`: the PubMed query retrieves 11 of those 13 papers (the two it misses, Vaishnav 2022 and Liao 2023, do not describe the assay in their abstracts). GEO is restricted to GSE Series (`gse[ETYP]`).

EDirect uses NCBI's current database index, so results can change between runs. Increase `--limit` after reviewing the sample. Increase EDirect's max search query result handling only if needed; the script first requests a bounded window of IDs and then retrieves those records.

## Table columns and screening guidance

The output includes a citation-style `short_name` (first author's surname and year, e.g. `De Boer et al. 2020`, matching the dataset names in `supptable.xlsx`; GEO Series take the short name of their linked paper and stay blank when NCBI links none), PMIDs, GSE accessions, title, authors/contact, year, journal, DOI, PubMed abstract or GEO summary, a deterministic extractive summary (up to three opening sentences), matched terms, heuristic relevance rating/reason, and links between records when EDirect has a link or an accession is explicitly mentioned. A `High` rating means a distinctive method term appeared in the title/abstract/summary; it is a prioritization aid, not a conclusion that the paper contains reusable MPRA data. Confirm methods, construct design, assay type, and supplementary files before inclusion. GEO and PubMed linkage is incomplete when NCBI has not connected records.

The script uses only Python's standard library and EDirect. Network access to NCBI is required when running a search.
