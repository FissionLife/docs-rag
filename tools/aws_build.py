"""Build an AWS service corpus from the 'Overview of Amazon Web Services' PDF.

Three stages, run in order:

    uv run python tools/aws_build.py parse      PDF markdown -> services.json
    uv run python tools/aws_build.py catalog    enrich from the AWS portal
    uv run python tools/aws_build.py pages      -> one .md per service

Stage 1 reads the whitepaper (converted to Markdown by markitdown) and recovers
the structure that the PDF-to-text conversion threw away. The converter emits no
headings at all -- just a flat stream with page furniture mixed in -- so the
document's own table of contents is used as a schema: it lists every category
and every service in reading order, which is exactly the segmentation the body
text needs and does not carry.

Stage 2 writes one Markdown page per service, which is the unit the RAG system
retrieves and cites.
"""
import json
import pathlib
import re
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
AWS = ROOT / "data" / "aws"
SOURCE_MD = AWS / "aws-overview.md"
SERVICES_JSON = AWS / "services.json"
PAGES_DIR = AWS / "services"

SOURCE_URL = ("https://docs.aws.amazon.com/whitepapers/latest/"
              "aws-overview/introduction.html")

# The AWS site backs its own product directory with this JSON endpoint. It is
# the same data the /products/ page renders, without the JavaScript -- one
# request for the whole catalogue instead of 350 HTML scrapes, and it carries
# fields the whitepaper does not have: the canonical product URL, AWS's own
# category, the pricing page, launch date and free-tier status.
CATALOG_API = (
    "https://aws.amazon.com/api/dirs/items/search"
    "?item.directoryId=aws-products"
    "&sort_by=item.additionalFields.productNameLowercase"
    "&sort_order=asc&size={size}&item.locale=en_US"
)
CATALOG_JSON = AWS / "catalog.json"

# Running header stamped on every page by the PDF layout.
RUNNING_HEADER = "Overview of Amazon Web Services AWS Whitepaper"

# TOC lines that are section headings rather than services. Everything else in
# the TOC is treated as a service, so this list only needs the exceptions the
# "starts with Amazon/AWS" test gets wrong in the other direction.
CATEGORIES = {
    "Analytics", "Application integration", "Blockchain",
    "Business applications", "Cloud Financial Management", "Compute",
    "Containers", "Customer enablement", "Databases", "Developer tools",
    "End user computing", "Frontend web and mobile services", "Game tech",
    "IoT", "ML and AI", "Management and governance", "Media",
    "Migration and transfer", "Networking and content delivery",
    "Quantum technologies", "Robotics", "Satellite",
    "Security, identity, and compliance", "Serverless", "Storage",
    "Accessing AWS services", "Global infrastructure",
    "Security and compliance", "Security", "Compliance",
    "Benefits of AWS security", "Introduction", "Abstract and introduction",
    "What is cloud computing?", "Six advantages of cloud computing",
    "Types of cloud computing", "Deployment models", "Cloud",
    "Private cloud (on-premises)", "Hybrid", "Contributors",
    "Document history", "AWS Glossary", "Further reading", "Conclusion",
}

# Services whose names do not begin with "Amazon"/"AWS".
NON_PREFIXED_SERVICES = {
    "Elastic Load Balancing", "VM Import/Export", "FreeRTOS", "Kiro",
    "OpsWorks", "VMware Cloud on AWS", "Red Hat OpenShift Service on AWS",
    "Savings Plans", "Reserved Instance (RI) reporting",
    "Migration Evaluator", "Snowball", "Snowcone", "Snowmobile",
}

# Sub-headings that sit *inside* a category. They are not services and must
# not become the current category -- "Compute" is followed by "Compare AWS
# compute services", and treating that as a category files EC2 under it.
IGNORE_HEADINGS = {
    "Compare AWS compute services", "Compare AWS database services",
    "Quick", "Connect Customer", "Note", "Resources",
}

_TOC_LINE = re.compile(r"^(.+?)\s*\.{4,}\s*(\d+)\s*$", re.M)


def _is_service(name: str) -> bool:
    if name in CATEGORIES or name in IGNORE_HEADINGS:
        return False
    if name in NON_PREFIXED_SERVICES:
        return True
    return name.startswith(("Amazon ", "AWS "))


def parse_toc(text: str) -> list[dict]:
    """Ordered [{name, category, page}] for every service the TOC lists."""
    out, category = [], "Uncategorised"
    seen = set()
    for name, page in _TOC_LINE.findall(text):
        name = re.sub(r"\s+", " ", name).strip()
        if not name or len(name) < 3:
            continue
        if _is_service(name):
            if name in seen:            # TOC repeats a few entries
                continue
            seen.add(name)
            out.append({"name": name, "category": category,
                        "page": int(page)})
        elif name in CATEGORIES:          # IGNORE_HEADINGS deliberately
            category = name               # leaves `category` untouched
    return out


def strip_furniture(body: str, names: set[str]) -> list[str]:
    """Drop running headers and page-number footers, return content lines."""
    lines = []
    for raw in body.split("\n"):
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue
        if line.strip() == RUNNING_HEADER:
            continue
        # Footers read "<current section name> <page>". Only drop them when the
        # prefix is a name we know, so real sentences ending in a number stay.
        m = re.fullmatch(r"(.+?)\s+(\d{1,3})", line.strip())
        if m and m.group(1).strip() in names:
            continue
        if re.fullmatch(r"[ivxlcdm]+|\d{1,3}", line.strip(), re.I):
            continue
        lines.append(line)
    return lines


def discover_body_services(lines: list[str], known: set[str]) -> list[dict]:
    """Find service sections the table of contents never listed.

    Blockchain, Game tech and Serverless have services in the body but no TOC
    entries for them, so a TOC-only schema silently drops them. The whitepaper
    opens almost every section by restating the service name -- "Amazon MQ" on
    its own line, then "Amazon MQ is a managed message broker..." -- and that
    restatement is a strong enough signal to recover the rest.
    """
    found, category = [], "Uncategorised"
    for i, line in enumerate(lines):
        name = line.strip()
        # Track the category headings as they go past, so a recovered service
        # lands in the section it actually sits in rather than a bucket.
        if name in CATEGORIES and name not in IGNORE_HEADINGS:
            category = name
            continue
        if (not name or name in known or len(name) > 60
                or not name.startswith(("Amazon ", "AWS "))):
            continue
        nxt = next((l.strip() for l in lines[i + 1:i + 3] if l.strip()), "")
        if nxt.startswith(name):
            found.append({"name": name, "category": category, "page": 0})
            known.add(name)
    return found


def segment(text: str, services: list[dict]) -> list[dict]:
    """Attach each service's body text, using service names as delimiters."""
    names = {s["name"] for s in services} | CATEGORIES
    # The TOC occupies the head of the document; the body restates every name.
    # Start after the last TOC line so TOC entries are not mistaken for
    # section starts.
    last_toc = max((m.end() for m in _TOC_LINE.finditer(text)), default=0)
    lines = strip_furniture(text[last_toc:], names)

    extra = discover_body_services(lines, set(names))
    if extra:
        print(f"recovered {len(extra)} service(s) absent from the TOC: "
              f"{', '.join(e['name'] for e in extra[:6])}"
              + (" ..." if len(extra) > 6 else ""))
        services = services + extra
        names |= {e["name"] for e in extra}

    index = {s["name"]: s for s in services}
    current, buf = None, []
    for line in lines:
        stripped = line.strip()
        if stripped in index:
            if current:
                current["text"] = "\n".join(buf).strip()
            current, buf = index[stripped], []
            continue
        if stripped in IGNORE_HEADINGS:
            continue
        if stripped in CATEGORIES and current:
            current["text"] = "\n".join(buf).strip()
            current, buf = None, []
            continue
        if current is not None:
            buf.append(line)
    if current:
        current["text"] = "\n".join(buf).strip()

    for s in services:
        s.setdefault("text", "")
        s["text"] = re.sub(r"\n{3,}", "\n\n", s["text"]).strip()
    return services


def _norm(name: str) -> str:
    """Loose key for matching whitepaper names against catalogue names."""
    n = name.lower()
    n = re.sub(r"(amazon|aws)", " ", n)
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n


def fetch_catalog() -> list[dict]:
    req = urllib.request.Request(
        CATALOG_API.format(size=1000),
        headers={"User-Agent": "docs-rag/1.0 (corpus builder)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))

    out = []
    for entry in data.get("items", []):
        f = entry.get("item", {}).get("additionalFields", {})
        name = (f.get("productName") or "").strip()
        if not name:
            continue
        out.append({
            "name": name,
            # Strip the campaign tracking parameters AWS appends.
            "url": (f.get("productUrl") or "").split("?")[0],
            "summary": re.sub(r"\s+", " ",
                              (f.get("productSummary") or "")).strip(),
            "category": (f.get("productCategory") or "").strip(),
            "pricing_url": (f.get("pricingUrl") or "").split("?")[0],
            "launched": (f.get("launchDate") or "").strip(),
            "free_tier": (f.get("freeTierAvailability") or "").strip(),
        })
    return out


def cmd_catalog() -> None:
    services = json.loads(SERVICES_JSON.read_text(encoding="utf-8"))
    print("fetching the AWS product directory ...")
    catalog = fetch_catalog()
    CATALOG_JSON.write_text(json.dumps(catalog, indent=1, ensure_ascii=False),
                            encoding="utf-8")
    print(f"  {len(catalog)} products -> {CATALOG_JSON}")

    by_key = {_norm(c["name"]): c for c in catalog}
    matched = 0
    for s in services:
        hit = by_key.get(_norm(s["name"]))
        if not hit:
            # Whitepaper often abbreviates ("Amazon EC2" vs "Amazon Elastic
            # Compute Cloud"); fall back to a containment match.
            key = _norm(s["name"])
            hit = next((c for k, c in by_key.items()
                        if key and (key in k or k in key)), None)
        if hit:
            matched += 1
            s["aws_url"] = hit["url"]
            s["aws_summary"] = hit["summary"]
            s["aws_category"] = hit["category"]
            s["pricing_url"] = hit["pricing_url"]
            s["launched"] = hit["launched"]
            s["free_tier"] = hit["free_tier"]

    SERVICES_JSON.write_text(json.dumps(services, indent=1,
                                        ensure_ascii=False), encoding="utf-8")
    print(f"  matched {matched}/{len(services)} whitepaper services to the "
          f"live catalogue")
    missing = [s["name"] for s in services if "aws_url" not in s]
    if missing:
        print(f"  {len(missing)} unmatched (retired or renamed): "
              f"{', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))


def cmd_parse() -> None:
    if not SOURCE_MD.exists():
        raise SystemExit(
            f"{SOURCE_MD} not found. Convert the PDF first:\n"
            '  uv tool run --python 3.13 --from "markitdown[pdf]" '
            'markitdown aws-overview.pdf -o data/aws/aws-overview.md')

    text = SOURCE_MD.read_text(encoding="utf-8")
    services = segment(text, parse_toc(text))

    kept = [s for s in services if len(s["text"]) >= 120]
    thin = [s for s in services if len(s["text"]) < 120]

    AWS.mkdir(parents=True, exist_ok=True)
    SERVICES_JSON.write_text(
        json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")

    cats = {}
    for s in kept:
        cats.setdefault(s["category"], []).append(s["name"])
    print(f"{len(kept)} services with descriptions across "
          f"{len(cats)} categories -> {SERVICES_JSON}")
    for c in sorted(cats):
        print(f"   {c:<38} {len(cats[c]):>3}")
    if thin:
        print(f"\n{len(thin)} entries had too little text and were dropped:")
        for s in thin[:15]:
            print(f"   {s['name']} ({len(s['text'])} chars)")


def cmd_pages() -> None:
    """One Markdown file per service -- the unit the RAG indexes and cites."""
    services = json.loads(SERVICES_JSON.read_text(encoding="utf-8"))
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in PAGES_DIR.glob("*.md"):
        f.unlink()

    for s in services:
        slug = re.sub(r"[^a-z0-9]+", "-", s["name"].lower()).strip("-")
        body = [f"# {s['name']}", ""]
        if s.get("aws_summary"):
            body += [f"> {s['aws_summary']}", ""]
        body += [f"**Category:** {s['category']}  "]
        if s.get("aws_category") and s["aws_category"] != s["category"]:
            body += [f"**AWS product category:** {s['aws_category']}  "]
        if s.get("launched"):
            body += [f"**Launched:** {s['launched']}  "]
        if s.get("free_tier"):
            body += [f"**Free tier:** {s['free_tier']}  "]
        if s.get("aws_url"):
            body += [f"**Product page:** {s['aws_url']}  "]
        if s.get("pricing_url"):
            body += [f"**Pricing:** {s['pricing_url']}  "]
        body += [f"**Source:** Overview of Amazon Web Services (AWS "
                 f"Whitepaper), page {s['page']} — {SOURCE_URL}", "",
                 "## Overview", "", s["text"], ""]
        (PAGES_DIR / f"{slug}.md").write_text("\n".join(body),
                                              encoding="utf-8")
    total = sum(len(s["text"]) for s in services)
    print(f"wrote {len(services)} service pages to {PAGES_DIR} "
          f"({total:,} chars of description)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "parse":
        cmd_parse()
    elif cmd == "catalog":
        cmd_catalog()
    elif cmd == "pages":
        cmd_pages()
    else:
        raise SystemExit(__doc__)
