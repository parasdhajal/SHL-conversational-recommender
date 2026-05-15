"""
Scrape SHL Talent Assessments catalog — Individual Test Solutions section only.

Follows paginated listing at /products/product-catalog/?start=...&type=1
(type=1 matches the site's catalog mode used in pagination links).

Outputs app/catalog.json with name, url, test_type, description, and optional columns.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlencode

import requests
from bs4 import BeautifulSoup

DEFAULT_CATALOG_BASE = "https://www.shl.com/products/product-catalog/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

PAGE_SIZE = 12
CATALOG_TYPE = "1"

TYPE_LETTERS: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _td_yes_no_circle(td: Any) -> str:
    """SHL uses empty spans like catalogue__circle -yes / -no for Remote and Adaptive columns."""
    if td is None:
        return ""
    yes = td.select_one("span.catalogue__circle.-yes")
    no = td.select_one("span.catalogue__circle.-no")
    if yes:
        return "Yes"
    if no:
        return "No"
    txt = _normalize_ws(td.get_text())
    return txt


def _parse_test_type_letters_from_td(td: Any) -> list[str]:
    letters: list[str] = []
    for sp in td.select("span.product-catalogue__key"):
        ch = _normalize_ws(sp.get_text())
        if len(ch) == 1 and ch.upper() in TYPE_LETTERS:
            u = ch.upper()
            if u not in letters:
                letters.append(u)
    if letters:
        return letters
    return _parse_test_type_letters(td.get_text())


def _parse_test_type_letters(cell_text: str) -> list[str]:
    raw = _normalize_ws(cell_text)
    if not raw:
        return []
    tokens = re.findall(r"[A-Za-z]", raw)
    letters: list[str] = []
    for ch in tokens:
        u = ch.upper()
        if u in TYPE_LETTERS and u not in letters:
            letters.append(u)
    return letters


def _letters_to_test_type(letters: list[str]) -> str:
    if not letters:
        return ""
    return "/".join(letters)


def _build_description(name: str, letters: list[str], remote: str, adaptive: str) -> str:
    parts = [name]
    if letters:
        expanded = ", ".join(f"{L} ({TYPE_LETTERS[L]})" for L in letters if L in TYPE_LETTERS)
        parts.append("Test types: " + expanded + ".")
    if remote:
        parts.append(f"Remote testing: {remote}.")
    if adaptive:
        parts.append(f"Adaptive/IRT: {adaptive}.")
    return _normalize_ws(" ".join(parts))


def _find_individual_test_table(soup: BeautifulSoup) -> BeautifulSoup | None:
    for th in soup.find_all("th"):
        text = _normalize_ws(th.get_text())
        if text.lower() == "individual test solutions":
            table = th.find_parent("table")
            if table:
                return table
    for hdr in soup.find_all(["h2", "h3", "h4", "strong", "p", "div"]):
        text = _normalize_ws(hdr.get_text())
        if text.lower() == "individual test solutions":
            nxt = hdr.find_next("table")
            if nxt:
                return nxt
    for table in soup.find_all("table"):
        prev = table.find_previous(string=re.compile(r"Individual\s+Test\s+Solutions", re.I))
        if prev:
            return table
    return None


def parse_catalog_html(html: str, page_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = _find_individual_test_table(soup)
    if not table:
        raise RuntimeError("Could not find Individual Test Solutions table — page structure may have changed.")

    items: list[dict[str, Any]] = []
    for row in table.find_all("tr"):
        if row.find_all("th"):
            continue
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
        first = tds[0]
        a = first.find("a", href=True)
        if not a:
            continue
        href = str(a["href"]).strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(page_url, href)
        if "/product-catalog/view/" not in urlparse(abs_url).path:
            continue
        name = _normalize_ws(a.get_text())
        if not name:
            continue
        remote = _td_yes_no_circle(tds[1]) if len(tds) > 1 else ""
        adaptive = _td_yes_no_circle(tds[2]) if len(tds) > 2 else ""
        type_td = tds[-1]
        letters = _parse_test_type_letters_from_td(type_td)
        test_type = _letters_to_test_type(letters)
        desc = _build_description(name, letters, remote, adaptive)
        items.append(
            {
                "name": name,
                "url": abs_url,
                "test_type": test_type,
                "description": desc,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
            }
        )

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for it in items:
        u = it["url"]
        if u in seen:
            continue
        seen.add(u)
        unique.append(it)
    return unique


def catalog_page_url(start: int, base: str) -> str:
    q = urlencode({"start": str(start), "type": CATALOG_TYPE})
    base = base.rstrip("/")
    return f"{base}/?{q}"


def scrape_individual_tests(
    base_url: str,
    sleep_s: float,
    timeout_s: float,
    max_pages: int = 50,
) -> list[dict[str, Any]]:
    sess = _session()
    all_by_url: dict[str, dict[str, Any]] = {}
    start = 0
    for _ in range(max_pages):
        page_url = catalog_page_url(start, base_url)
        r = sess.get(page_url, timeout=timeout_s)
        r.raise_for_status()
        time.sleep(sleep_s)
        batch = parse_catalog_html(r.text, page_url)
        new = 0
        for it in batch:
            if it["url"] not in all_by_url:
                all_by_url[it["url"]] = it
                new += 1
        if new == 0:
            break
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return list(all_by_url.values())


def main() -> int:
    p = argparse.ArgumentParser(description="Scrape SHL Individual Test Solutions into app/catalog.json")
    p.add_argument(
        "--base-url",
        default=DEFAULT_CATALOG_BASE,
        help="Catalog listing base (pagination uses ?start=&type=)",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "app" / "catalog.json"),
        help="Output JSON path",
    )
    p.add_argument("--sleep", type=float, default=0.45, help="Politeness delay after each HTTP GET (seconds)")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout (seconds)")
    p.add_argument("--max-pages", type=int, default=50, help="Safety cap on pagination depth")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = scrape_individual_tests(
        base_url=args.base_url,
        sleep_s=args.sleep,
        timeout_s=args.timeout,
        max_pages=args.max_pages,
    )
    items.sort(key=lambda x: x["name"].lower())
    if not items:
        print("No items parsed; aborting.", file=sys.stderr)
        return 2

    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} catalog entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
