"""Turn any source into a document the pipeline understands.

Every loader returns the same shape:

    {"doc_id": str, "title": str, "url": str, "text": str, "kind": str}

`text` is normalised to lightweight Markdown, because `chunk.py` splits on
headings and headings are the single most useful structural signal a document
can carry. Where a format has no headings (plain text, most PDFs) we
synthesise the best boundary available -- page numbers for PDFs -- so that
citations can still say *where* in the document a passage came from.

Supported:

    .pdf                      pypdf, one section per page
    .md .markdown .mdx        as-is; ATX (#) headings already work
    .txt .text .rst .log      plain text, no sections
    .html .htm                trafilatura article extraction -> Markdown
    http(s)://...             fetched, then as above
    en.wikipedia.org/wiki/X   MediaWiki API (cleaner than scraping the page)
    a directory               every supported file inside it, recursively

Why trafilatura rather than hand-rolled HTML stripping: a Medium or blog page
is mostly navigation, related-post rails, cookie banners and footers. Feeding
that into the index poisons retrieval with text nobody asked about. Boilerplate
removal is a genuinely hard problem and not worth re-solving badly.
"""
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

UA = "docs-rag/1.0 (personal RAG indexer)"

TEXT_EXT = {".txt", ".text", ".rst", ".log", ".csv"}
MD_EXT = {".md", ".markdown", ".mdx"}
HTML_EXT = {".html", ".htm", ".xhtml"}
PDF_EXT = {".pdf"}
SUPPORTED = TEXT_EXT | MD_EXT | HTML_EXT | PDF_EXT


class LoadError(RuntimeError):
    """A source could not be read or produced no usable text."""


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:n].rstrip("-") or "doc")


def _doc_id(title: str, source: str) -> str:
    """Stable, unique, readable. The hash keeps same-titled sources apart."""
    h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:6]
    return f"{_slug(title)}-{h}"


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # runs of blank lines
    return text.strip()


# --------------------------------------------------------------- Wikipedia

def _wikipedia(url: str) -> dict:
    """Use the MediaWiki API: the rendered page is full of chrome, the API
    gives clean plaintext with '== Heading ==' markers already in it."""
    m = re.search(r"//([a-z-]+)\.wikipedia\.org/wiki/([^?#]+)", url)
    if not m:
        raise LoadError(f"not a recognisable Wikipedia article URL: {url}")
    lang, title = m.group(1), urllib.parse.unquote(m.group(2)).replace("_", " ")

    api = (f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts", "explaintext": "1", "exlimit": "1",
        "redirects": "1", "titles": title}))
    req = urllib.request.Request(api, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise LoadError(f"Wikipedia has no article '{title}'")
    page = pages[0]
    text = _strip_wiki_boilerplate(page.get("extract", ""))
    if len(text) < 200:
        raise LoadError(f"Wikipedia article '{title}' had almost no text")

    canonical = (f"https://{lang}.wikipedia.org/wiki/"
                 + urllib.parse.quote(page["title"].replace(" ", "_")))
    return {"doc_id": _doc_id(page["title"], canonical), "title": page["title"],
            "url": canonical, "text": _clean(text), "kind": "wikipedia"}


_DROP = {"see also", "references", "further reading", "external links",
         "notes", "bibliography", "sources", "citations"}


def _strip_wiki_boilerplate(text: str) -> str:
    """Drop reference/see-also sections: no answerable content, lots of noise."""
    out, skipping = [], False
    for line in text.split("\n"):
        m = re.fullmatch(r"\s*(={2,})\s*(.+?)\s*\1\s*", line)
        if m and len(m.group(1)) == 2:      # only level-2 headings decide
            skipping = m.group(2).strip().lower() in _DROP
        if not skipping:
            out.append(line)
    return "\n".join(out)


# -------------------------------------------------------------------- HTML

def _from_html(html: str, url: str, fallback_title: str) -> dict:
    import trafilatura

    text = trafilatura.extract(
        html, output_format="markdown", include_comments=False,
        include_tables=True, include_formatting=True, favor_precision=True)
    if not text or len(text.strip()) < 200:
        # favor_precision drops too much on sparse pages; retry permissively.
        text = trafilatura.extract(
            html, output_format="markdown", include_comments=False,
            include_tables=True, favor_recall=True)
    if not text or len(text.strip()) < 100:
        raise LoadError(
            f"no article text could be extracted from {url or fallback_title}. "
            "The page may be JavaScript-rendered or paywalled; save it as "
            "HTML or PDF and pass the file instead.")

    title = fallback_title
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            title = meta.title.strip()
    except Exception:
        pass

    return {"doc_id": _doc_id(title, url or title), "title": title,
            "url": url, "text": _clean(text), "kind": "html"}


# ------------------------------------------------------------------ GitHub

_GH_BLOB = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)(?:\?[^#]*)?(?:#.*)?$")
_GH_REPO = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/?$")
_GH_TREE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/tree/")


def _fetch_text(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _from_github(url: str) -> dict | None:
    """Fetch GitHub-hosted files as raw source rather than scraping the page.

    The rendered blob page is an application, not a document: scraping it
    yields the prose but loses every heading, because GitHub emits <h2> inside
    its own layout and boilerplate removal cannot tell the file's headings from
    the page's chrome. Losing headings means losing section breadcrumbs, which
    is most of what makes a chunk retrievable. raw.githubusercontent.com serves
    the bytes the author wrote, so Markdown stays Markdown.
    """
    if _GH_TREE.match(url):
        raise LoadError(
            "that is a directory listing. Link a specific file (a /blob/ URL), "
            "or clone the repo and pass the folder path.")

    m = _GH_BLOB.match(url)
    if m:
        owner, repo, ref, path = m.groups()
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        text = _fetch_text(raw)
        if text is None:
            raise LoadError(f"could not fetch {raw}")
        return _make_github_doc(text, path.rsplit("/", 1)[-1], url)

    m = _GH_REPO.match(url)
    if m:
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        for ref in ("main", "master"):
            for name in ("README.md", "readme.md", "README.rst", "README.txt"):
                raw = (f"https://raw.githubusercontent.com/"
                       f"{owner}/{repo}/{ref}/{name}")
                text = _fetch_text(raw)
                if text:
                    return _make_github_doc(text, name, url)
        raise LoadError(f"no README found in {owner}/{repo} on main or master")

    if url.startswith("https://raw.githubusercontent.com/"):
        text = _fetch_text(url)
        if text is None:
            raise LoadError(f"could not fetch {url}")
        return _make_github_doc(text, url.rsplit("/", 1)[-1], url)

    return None          # some other github.com page -- scrape it normally


def _make_github_doc(text: str, filename: str, url: str) -> dict:
    if not text.strip():
        raise LoadError(f"{filename} is empty")
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    kind = "markdown" if ext in MD_EXT else "text"
    title = _md_title(text, filename) if ext in MD_EXT else filename
    return {"doc_id": _doc_id(title, url), "title": title, "url": url,
            "text": _clean(text), "kind": kind}


def _from_url(url: str) -> dict:
    if re.search(r"//[a-z-]+\.wikipedia\.org/wiki/", url):
        return _wikipedia(url)

    if re.match(r"^https?://(github\.com|raw\.githubusercontent\.com)/", url):
        doc = _from_github(url)
        if doc is not None:
            return doc

    import trafilatura
    html = trafilatura.fetch_url(url)
    if not html:
        raise LoadError(f"could not fetch {url}")
    return _from_html(html, url, urllib.parse.urlparse(url).path or url)


# --------------------------------------------------------------------- PDF

def _from_pdf(path: Path) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")          # many PDFs are "encrypted" with no password
        except Exception:
            raise LoadError(f"{path.name} is password-protected")

    # A PDF has no headings to split on, so pages become the sections. That
    # keeps citations useful: "Report > Page 12" beats "Report" alone.
    parts, empty = [], 0
    for i, page in enumerate(reader.pages, 1):
        try:
            t = (page.extract_text() or "").strip()
        except Exception:
            t = ""
        if not t:
            empty += 1
            continue
        parts.append(f"## Page {i}\n\n{t}")

    if not parts:
        raise LoadError(
            f"{path.name} yielded no extractable text ({len(reader.pages)} "
            "pages). It is probably a scan -- run OCR on it first.")
    if empty:
        print(f"    note: {empty}/{len(reader.pages)} pages had no text layer")

    title = path.stem
    try:
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title.strip() or path.stem
    except Exception:
        pass

    return {"doc_id": _doc_id(title, str(path.resolve())), "title": title,
            "url": path.resolve().as_uri(), "text": _clean("\n\n".join(parts)),
            "kind": "pdf"}


# ------------------------------------------------------------- plain / file

def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise LoadError(f"could not decode {path.name} as text")


_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.S)


def split_front_matter(text: str) -> tuple[dict, str]:
    """Pull a leading `key: value` block off a Markdown file.

    Generated pages know where their content really came from, but the loader
    would otherwise label them with the local file path -- so a citation reads
    `file:///C:/.../amazon-s3.md` instead of the page a reader can open. A
    `source_url:` in front matter lets the generator declare the canonical URL
    and have citations point at it.

    Deliberately not YAML: this reads flat `key: value` lines only, so it
    needs no parser and cannot execute anything from a document.
    """
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[m.end():]


def _md_title(text: str, default: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else default


def _from_file(path: Path) -> dict:
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return _from_pdf(path)

    raw = _read_text(path)
    if not raw.strip():
        raise LoadError(f"{path.name} is empty")

    if ext in HTML_EXT:
        return _from_html(raw, path.resolve().as_uri(), path.stem)

    meta = {}
    if ext in MD_EXT:
        meta, raw = split_front_matter(raw)

    title = meta.get("title") or (
        _md_title(raw, path.stem) if ext in MD_EXT else path.stem)
    url = meta.get("source_url") or path.resolve().as_uri()
    return {"doc_id": _doc_id(title, str(path.resolve())), "title": title,
            "url": url, "text": _clean(raw),
            "kind": "markdown" if ext in MD_EXT else "text"}


# ------------------------------------------------------------------ public

def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def load_source(source: str) -> list[dict]:
    """Load one source. A directory expands to every supported file inside."""
    if is_url(source):
        return [_from_url(source)]

    path = Path(source).expanduser()
    if not path.exists():
        raise LoadError(f"no such file or directory: {source}")

    if path.is_dir():
        files = sorted(p for p in path.rglob("*")
                       if p.is_file() and p.suffix.lower() in SUPPORTED)
        if not files:
            raise LoadError(
                f"{path} contains no supported files "
                f"({', '.join(sorted(SUPPORTED))})")
        docs = []
        for f in files:
            try:
                docs.append(_from_file(f))
            except LoadError as e:
                print(f"    skipped {f.name}: {e}")
        if not docs:
            raise LoadError(f"nothing in {path} could be read")
        return docs

    if path.suffix.lower() not in SUPPORTED:
        raise LoadError(
            f"unsupported file type '{path.suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED))}")
    return [_from_file(path)]
