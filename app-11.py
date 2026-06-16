import os
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

API_BASE = "https://api.company-information.service.gov.uk"
TECH_SIC_CODES = ["62012", "72110"]
CACHE_FILE = "screened_companies_cache.json"

# PSC kind prefixes that indicate a corporate/legal entity (i.e. a company)
CORPORATE_KINDS = (
    "corporate-entity-person-with-significant-control",
    "corporate-entity-beneficial-owner",
    "legal-person-with-significant-control",
    "legal-person-beneficial-owner",
)

# PSC kind prefixes that indicate a natural person (i.e. a founder/individual)
INDIVIDUAL_KINDS = (
    "individual-person-with-significant-control",
    "individual-beneficial-owner",
)

# natures_of_control values that indicate 25%+ share ownership
SHARE_OWNERSHIP_NATURES = (
    "ownership-of-shares-25-to-50-percent",
    "ownership-of-shares-50-to-75-percent",
    "ownership-of-shares-75-to-100-percent",
    "ownership-of-shares-25-to-50-percent-as-trust",
    "ownership-of-shares-50-to-75-percent-as-trust",
    "ownership-of-shares-75-to-100-percent-as-trust",
    "ownership-of-shares-25-to-50-percent-as-firm",
    "ownership-of-shares-50-to-75-percent-as-firm",
    "ownership-of-shares-75-to-100-percent-as-firm",
    "ownership-of-shares-more-than-25-percent-registered-overseas-entity",
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> Dict:
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {"screened": {}, "hits": []}


def save_cache(cache: Dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def mark_screened(cache: Dict, company_number: str, rows: List[Dict]) -> None:
    cache["screened"][company_number] = True
    if rows:
        existing = {(r["company_number"], r["founder_name"], r["acquirer_company_name"]) for r in cache["hits"]}
        for row in rows:
            key = (row["company_number"], row["founder_name"], row["acquirer_company_name"])
            if key not in existing:
                cache["hits"].append(row)
                existing.add(key)


# ---------------------------------------------------------------------------
# Companies House API client
# ---------------------------------------------------------------------------

class CompaniesHouseClient:
    def __init__(self, api_key: str, sleep_seconds: float = 0.2):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.session.headers.update({"User-Agent": "streamlit-liquidity-event-screener/1.0"})

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{API_BASE}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        time.sleep(self.sleep_seconds)
        return resp.json()

    def advanced_company_search(self, sic_codes: List[str], size: int = 100, start_index: int = 0) -> Dict:
        return self._get("/advanced-search/companies", params={
            "size": size,
            "start_index": start_index,
            "sic_codes": ",".join(sic_codes),
            "company_status": "active",
        })

    def get_company_profile(self, company_number: str) -> Dict:
        return self._get(f"/company/{company_number}")

    def get_filing_history(self, company_number: str, items_per_page: int = 100) -> Dict:
        return self._get(f"/company/{company_number}/filing-history",
                         params={"items_per_page": items_per_page})

    def list_pscs(self, company_number: str) -> List[Dict]:
        return self._get(f"/company/{company_number}/persons-with-significant-control").get("items", [])


# ---------------------------------------------------------------------------
# PSC helpers
# ---------------------------------------------------------------------------

def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def has_share_ownership_25plus(psc: Dict) -> bool:
    """Return True if the PSC holds 25%+ of shares."""
    natures = psc.get("natures_of_control") or []
    return any(n in SHARE_OWNERSHIP_NATURES for n in natures)


def extract_control_band(natures: List[str]) -> str:
    text = " | ".join(natures or [])
    if "75-to-100" in text or "75-to-100-percent" in text:
        return "75–100%"
    if "50-to-75" in text:
        return "50–75%"
    if "25-to-50" in text:
        return "25–50%"
    if "more-than-25" in text:
        return ">25%"
    return "Unknown"


def is_individual_psc(psc: Dict) -> bool:
    return (psc.get("kind") or "").startswith(INDIVIDUAL_KINDS)


def is_corporate_psc(psc: Dict) -> bool:
    kind = psc.get("kind") or ""
    return any(kind.startswith(k) for k in CORPORATE_KINDS)


def psc_is_founder_like(psc: Dict, incorporation_date: Optional[datetime]) -> bool:
    """
    An individual is 'founder-like' if they were notified as a PSC within
    2 years of incorporation OR if their PSC record has since been ceased
    (meaning they've exited).
    """
    if psc.get("ceased_on"):
        return True
    notified = parse_date(psc.get("notified_on"))
    if incorporation_date and notified:
        return abs((notified - incorporation_date).days) <= 730
    return False


# ---------------------------------------------------------------------------
# Core signal detection — the new strict logic
# ---------------------------------------------------------------------------

def find_corporate_acquirers(pscs: List[Dict], incorporation_date: Optional[datetime]) -> List[Dict]:
    """
    Return corporate/legal-entity PSCs that:
    1. Hold 25%+ of shares.
    2. Were notified AFTER the company was incorporated (i.e. they came in later,
       not as an original shareholder — ruling out holding-company structures
       set up at inception).
    3. Are currently active (not ceased) — meaning they still hold the shares,
       i.e. the acquisition happened and they stayed in.
    """
    acquirers = []
    for psc in pscs:
        if not is_corporate_psc(psc):
            continue
        if not has_share_ownership_25plus(psc):
            continue
        if psc.get("ceased_on"):
            # They came and went — less useful, but still worth flagging
            pass
        notified = parse_date(psc.get("notified_on"))
        if incorporation_date and notified:
            days_after = (notified - incorporation_date).days
            if days_after < 30:
                # Registered on or right at incorporation — original structure, skip
                continue
        acquirers.append(psc)
    return acquirers


def find_founder_individuals(pscs: List[Dict], incorporation_date: Optional[datetime]) -> List[Dict]:
    """
    Return individual PSCs that are founder-like AND held 25%+ of shares.
    """
    founders = []
    for psc in pscs:
        if not is_individual_psc(psc):
            continue
        if not has_share_ownership_25plus(psc):
            continue
        if psc_is_founder_like(psc, incorporation_date):
            founders.append(psc)
    return founders


def extract_capital_from_filings(filings: List[Dict]) -> str:
    for item in filings:
        desc = (item.get("description") or "").lower()
        if "statement of capital" in desc or "share allotment" in desc or "sh01" in (item.get("type") or "").lower():
            m = re.search(r"([£$€]\s?[\d,]+(?:\.\d+)?)", desc, re.I)
            if m:
                return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Analyse a single company
# ---------------------------------------------------------------------------

def is_active(profile: Dict) -> bool:
    return (profile.get("company_status") or "").lower() == "active"


def company_within_years(profile: Dict, years_back: int) -> bool:
    created = parse_date(profile.get("date_of_creation"))
    if not created:
        return False
    return (datetime.now().year - created.year) <= years_back


def analyse_company(client: CompaniesHouseClient, company_item: Dict, years_back: int) -> List[Dict]:
    company_number = company_item.get("company_number")
    profile = client.get_company_profile(company_number)

    if not is_active(profile):
        return []
    if not company_within_years(profile, years_back):
        return []

    incorporation_date = parse_date(profile.get("date_of_creation"))
    pscs = client.list_pscs(company_number)

    # ---- THE CORE SIGNAL: a company entered as 25%+ shareholder post-incorporation ----
    acquirers = find_corporate_acquirers(pscs, incorporation_date)
    if not acquirers:
        return []  # No corporate entity has bought in — skip entirely

    # ---- FOUNDERS: individuals with 25%+ share ownership at or near inception ----
    founders = find_founder_individuals(pscs, incorporation_date)
    if not founders:
        return []  # No identifiable individual founder — skip

    filings = client.get_filing_history(company_number).get("items", [])
    capital = extract_capital_from_filings(filings)
    ch_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"

    rows = []
    for acquirer in acquirers:
        acquirer_name = acquirer.get("name") or acquirer.get("identification", {}).get("legal_name") or "Unknown"
        acquirer_notified = acquirer.get("notified_on") or ""
        acquirer_ceased = acquirer.get("ceased_on") or ""
        acquirer_band = extract_control_band(acquirer.get("natures_of_control") or [])
        acquirer_country = acquirer.get("identification", {}).get("country_registered") or acquirer.get("address", {}).get("country") or ""

        for founder in founders:
            address = founder.get("address") or {}
            dob = founder.get("date_of_birth") or {}
            rows.append({
                "target_company_name": profile.get("company_name"),
                "company_number": company_number,
                "sic_codes": ", ".join(profile.get("sic_codes") or []),
                "date_of_creation": profile.get("date_of_creation") or "",
                "founder_name": founder.get("name") or "Unknown",
                "founder_postcode": address.get("postal_code") or "",
                "founder_dob": f"{dob.get('month','')}/{dob.get('year','')}".strip("/"),
                "founder_control_band": extract_control_band(founder.get("natures_of_control") or []),
                "founder_ceased_on": founder.get("ceased_on") or "",
                "acquirer_company_name": acquirer_name,
                "acquirer_notified_on": acquirer_notified,
                "acquirer_ceased_on": acquirer_ceased,
                "acquirer_control_band": acquirer_band,
                "acquirer_country": acquirer_country,
                "capital_injected": capital,
                "company_profile_link": ch_url,
            })
    return rows


# ---------------------------------------------------------------------------
# Main screening loop
# ---------------------------------------------------------------------------

def get_all_target_companies(client: CompaniesHouseClient, sic_codes: List[str], max_companies: int) -> List[Dict]:
    companies, start_index, page_size = [], 0, 100
    while len(companies) < max_companies:
        data = client.advanced_company_search(sic_codes=sic_codes, size=page_size, start_index=start_index)
        items = data.get("items") or []
        if not items:
            break
        companies.extend(items)
        start_index += page_size
        if len(items) < page_size:
            break
    return companies[:max_companies]


def run_screen(api_key: str, max_companies: int, years_back: int, sleep_seconds: float) -> Tuple[pd.DataFrame, Dict]:
    client = CompaniesHouseClient(api_key=api_key, sleep_seconds=sleep_seconds)
    cache = load_cache()

    companies = get_all_target_companies(client, TECH_SIC_CODES, max_companies=max_companies)
    total_matched = len(companies)

    already_screened = sum(1 for c in companies if c.get("company_number") in cache["screened"])
    to_screen = [c for c in companies if c.get("company_number") not in cache["screened"]]

    if to_screen:
        progress_bar = st.progress(0, text="Starting scan…")
        for i, company in enumerate(to_screen):
            try:
                rows = analyse_company(client, company, years_back)
                mark_screened(cache, company["company_number"], rows)
            except Exception:
                mark_screened(cache, company["company_number"], [])
            pct = int((i + 1) / len(to_screen) * 100)
            progress_bar.progress(pct, text=f"Screened {i + 1} of {len(to_screen)} new companies…")
        progress_bar.empty()
        save_cache(cache)
    else:
        st.info("All companies in the current result set are already cached. No new screening needed.")

    all_hits = cache.get("hits") or []
    df = pd.DataFrame(all_hits) if all_hits else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["company_number", "founder_name", "acquirer_company_name"])
        df = df.sort_values(by=["acquirer_notified_on", "date_of_creation"], ascending=[False, False], na_position="last")

    stats = {
        "total_matched": total_matched,
        "already_screened": already_screened,
        "newly_screened": len(to_screen),
        "total_screened": len(cache["screened"]),
        "total_hits": len(df) if not df.empty else 0,
    }
    return df, stats


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CH Liquidity Event Screener", layout="wide")
st.title("Companies House Liquidity Event Screener")
st.caption(
    "Identifies UK tech & biotech companies (SIC 62012 / 72110) where a corporate entity "
    "has acquired 25%+ of shares from a founder post-incorporation."
)

with st.sidebar:
    st.header("Configuration")
    _default_key = (st.secrets.get("companies_house") or {}).get("api_key", "") or os.getenv("CH_API_KEY", "")
    api_key = st.text_input("Companies House API key", type="password", value=_default_key)
    max_companies = st.slider("Max companies to fetch", min_value=50, max_value=1000, value=250, step=50)
    years_back = st.slider("Founded within last N years", min_value=1, max_value=15, value=10)
    sleep_seconds = st.slider("API throttle (secs per call)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.divider()
    if st.button("🗑️ Clear cache & re-screen all", type="secondary"):
        if Path(CACHE_FILE).exists():
            Path(CACHE_FILE).unlink()
        st.cache_data.clear()
        st.success("Cache cleared.")
    run = st.button("▶ Run screen", type="primary")

with st.expander("ℹ️ How signals are detected", expanded=False):
    st.markdown("""
**A match requires ALL of the following to be true:**

1. The company is **active** and incorporated within the selected window.
2. At least one **individual PSC** holds 25%+ of shares and was notified within 2 years of incorporation (founder-like), OR has since ceased (exited).
3. At least one **corporate-entity PSC** (a company) holds 25%+ of shares AND was notified **more than 30 days after incorporation** — ruling out holding structures set up at day one.

**What is reported per row:**
- The founder's name, partial DOB, postcode, and control band.
- The acquiring company's name, country, control band, and date they were notified.
- Capital injected (where inferable from SH01/statement of capital filings).
- A link to the Companies House profile.

**Limitations:** PSC cessation does not always accompany a sale. Capital figures are inferred from filing descriptions only and are often absent.
    """)

if run:
    if not api_key:
        st.error("Please enter your Companies House API key in the sidebar.")
    else:
        with st.spinner("Fetching company list…"):
            df, stats = run_screen(api_key, max_companies, years_back, sleep_seconds)

        st.subheader("Screening summary")
        cols = st.columns(5)
        cols[0].metric("Companies matched (SIC)", stats["total_matched"])
        cols[1].metric("Already in cache", stats["already_screened"])
        cols[2].metric("Newly screened", stats["newly_screened"])
        cols[3].metric("Total screened to date", stats["total_screened"])
        cols[4].metric("🎯 Total hits", stats["total_hits"])

        st.divider()

        if df.empty:
            st.warning("No corporate acquisition signals found yet. Try increasing the company fetch limit.")
        else:
            st.success(f"{len(df)} founder/acquirer rows found across {df['company_number'].nunique()} companies.")

            display_df = df.copy()
            display_df["company_profile_link"] = display_df["company_profile_link"].apply(
                lambda u: f'<a href="{u}" target="_blank">Here</a>' if u else ""
            )
            display_df = display_df.rename(columns={
                "target_company_name":   "Company",
                "company_number":        "Co. No.",
                "sic_codes":             "SIC",
                "date_of_creation":      "Incorporated",
                "founder_name":          "Founder",
                "founder_postcode":      "Postcode",
                "founder_dob":           "DOB (m/y)",
                "founder_control_band":  "Founder Band",
                "founder_ceased_on":     "Founder Ceased",
                "acquirer_company_name": "Acquirer",
                "acquirer_notified_on":  "Acquirer Notified",
                "acquirer_ceased_on":    "Acquirer Ceased",
                "acquirer_control_band": "Acquirer Band",
                "acquirer_country":      "Acquirer Country",
                "capital_injected":      "Capital",
                "company_profile_link":  "CH Profile",
            })

            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download results as CSV",
                data=csv_data,
                file_name="ch_liquidity_screen.csv",
                mime="text/csv",
            )
        for row in rows:
            if row["company_number"] not in existing_numbers:
                cache["hits"].append(row)
                existing_numbers.add(row["company_number"])


# ---------------------------------------------------------------------------
# Companies House API client
# ---------------------------------------------------------------------------

class CompaniesHouseClient:
    def __init__(self, api_key: str, sleep_seconds: float = 0.2):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.session.headers.update({"User-Agent": "streamlit-liquidity-event-screener/1.0"})

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{API_BASE}{path}"
        response = self.session.get(url, params=params, timeout=30)
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        time.sleep(self.sleep_seconds)
        return response.json()

    def advanced_company_search(self, sic_codes: List[str], size: int = 100, start_index: int = 0) -> Dict:
        params = {
            "size": size,
            "start_index": start_index,
            "sic_codes": ",".join(sic_codes),
            "company_status": "active",
        }
        return self._get("/advanced-search/companies", params=params)

    def get_company_profile(self, company_number: str) -> Dict:
        return self._get(f"/company/{company_number}")

    def get_filing_history(self, company_number: str, items_per_page: int = 100) -> Dict:
        return self._get(
            f"/company/{company_number}/filing-history",
            params={"items_per_page": items_per_page},
        )

    def list_pscs(self, company_number: str) -> Dict:
        return self._get(f"/company/{company_number}/persons-with-significant-control")

    def get_psc_details(self, company_number: str, item: Dict) -> Dict:
        links = item.get("links", {}).get("self", "")
        if not links:
            return {}
        path = links if links.startswith("/") else "/" + links
        return self._get(path)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

LIQUIDITY_PATTERNS = [
    r"share allotment", r"statement of capital", r"cessation",
    r"person with significant control", r"transfer", r"acquisition",
]

PURCHASER_HINTS = [
    r"purchased by ([A-Za-z0-9&.,'()\- ]+)",
    r"acquired by ([A-Za-z0-9&.,'()\- ]+)",
    r"allotted to ([A-Za-z0-9&.,'()\- ]+)",
    r"issued to ([A-Za-z0-9&.,'()\- ]+)",
]


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def company_within_years(company: Dict, years_back: int) -> bool:
    created = parse_date(company.get("date_of_creation"))
    if not created:
        return False
    return (datetime.now().year - created.year) <= years_back


def is_active(company: Dict) -> bool:
    status = (company.get("company_status") or "").lower()
    return status == "active"


def psc_is_founder_like(psc: Dict, incorporation_date: Optional[datetime]) -> bool:
    if psc.get("ceased_on"):
        return True
    notified = parse_date(psc.get("notified_on"))
    if incorporation_date and notified:
        return abs((notified - incorporation_date).days) <= 730
    return False


def extract_control_band(natures: List[str]) -> str:
    text = " | ".join(natures or [])
    if "75-to-100" in text: return "75–100%"
    if "50-to-75" in text:  return "50–75%"
    if "25-to-50" in text:  return "25–50%"
    return "Unknown"


def scan_filings(filings: List[Dict]) -> Tuple[bool, str, str, str]:
    matched, purchaser, capital = [], "", ""
    for item in filings:
        desc = f"{item.get('category','')} {item.get('type','')} {item.get('description','')}".lower()
        if any(re.search(p, desc) for p in LIQUIDITY_PATTERNS):
            matched.append(f"{item.get('date','unknown')}: {item.get('type','')} – {item.get('description','')}")
            if not capital:
                m = re.search(r"([£$€]\s?[\d,]+(?:\.\d+)?)", desc, re.I)
                if m: capital = m.group(1)
            if not purchaser:
                for p in PURCHASER_HINTS:
                    m = re.search(p, item.get("description", ""), re.I)
                    if m:
                        purchaser = m.group(1).strip(" .,")
                        break
    return bool(matched), " ; ".join(matched[:8]), purchaser, capital


def build_founder_rows(client: CompaniesHouseClient, company_number: str, incorporation_date: Optional[datetime]) -> List[Dict]:
    founders = []
    for item in client.list_pscs(company_number).get("items", []):
        detail = client.get_psc_details(company_number, item) or item
        natures = detail.get("natures_of_control") or item.get("natures_of_control") or []
        if not any("ownership-of-shares" in n for n in natures):
            continue
        if not psc_is_founder_like(detail, incorporation_date):
            continue
        address = detail.get("address", {}) or {}
        dob = detail.get("date_of_birth", {}) or {}
        founders.append({
            "founder_name": detail.get("name") or item.get("name") or "Unknown",
            "founder_postcode": address.get("postal_code", ""),
            "founder_dob": f"{dob.get('month','')}/{dob.get('year','')}".strip("/"),
            "founder_control_band": extract_control_band(natures),
            "founder_ceased_on": detail.get("ceased_on") or item.get("ceased_on") or "",
            "founder_notified_on": detail.get("notified_on") or item.get("notified_on") or "",
        })
    return founders


def analyse_company(client: CompaniesHouseClient, company_item: Dict, years_back: int) -> List[Dict]:
    company_number = company_item.get("company_number")
    profile = client.get_company_profile(company_number)

    if not is_active(profile):
        return []
    if not company_within_years(profile, years_back):
        return []

    incorporation_date = parse_date(profile.get("date_of_creation"))
    filings = client.get_filing_history(company_number).get("items", [])
    has_signal, signal_text, purchaser_name, capital = scan_filings(filings)
    founders = build_founder_rows(client, company_number, incorporation_date)

    ch_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
    output = []
    for founder in founders:
        if founder["founder_ceased_on"] or has_signal:
            output.append({
                "target_company_name": profile.get("company_name"),
                "company_number": company_number,
                "sic_codes": ", ".join(profile.get("sic_codes", [])),
                "date_of_creation": profile.get("date_of_creation", ""),
                "founder_name": founder["founder_name"],
                "founder_postcode": founder["founder_postcode"],
                "founder_dob": founder["founder_dob"],
                "founder_control_band": founder["founder_control_band"],
                "founder_ceased_on": founder["founder_ceased_on"],
                "capital_injected": capital,
                "purchaser_company_name": purchaser_name,
                "liquidity_signal": signal_text,
                "company_profile_link": ch_url,
            })
    return output


# ---------------------------------------------------------------------------
# Main screening loop
# ---------------------------------------------------------------------------

def get_all_target_companies(client: CompaniesHouseClient, sic_codes: List[str], max_companies: int) -> List[Dict]:
    companies, start_index, page_size = [], 0, 100
    while len(companies) < max_companies:
        data = client.advanced_company_search(sic_codes=sic_codes, size=page_size, start_index=start_index)
        items = data.get("items", [])
        if not items:
            break
        companies.extend(items)
        start_index += page_size
        if len(items) < page_size:
            break
    return companies[:max_companies]


def run_screen(api_key: str, max_companies: int, years_back: int, sleep_seconds: float) -> Tuple[pd.DataFrame, Dict]:
    client = CompaniesHouseClient(api_key=api_key, sleep_seconds=sleep_seconds)
    cache = load_cache()

    companies = get_all_target_companies(client, TECH_SIC_CODES, max_companies=max_companies)
    total_matched = len(companies)

    already_screened = sum(1 for c in companies if c.get("company_number") in cache["screened"])
    to_screen = [c for c in companies if c.get("company_number") not in cache["screened"]]

    progress_bar = st.progress(0, text="Starting scan…")
    new_rows = 0

    for i, company in enumerate(to_screen):
        try:
            rows = analyse_company(client, company, years_back)
            mark_screened(cache, company["company_number"], rows)
            if rows:
                new_rows += len(rows)
        except Exception:
            mark_screened(cache, company["company_number"], [])
        pct = int((i + 1) / len(to_screen) * 100) if to_screen else 100
        progress_bar.progress(pct, text=f"Screened {i+1} of {len(to_screen)} new companies…")

    save_cache(cache)
    progress_bar.empty()

    all_hits = cache.get("hits", [])
    df = pd.DataFrame(all_hits) if all_hits else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["company_number", "founder_name"])
        df = df.sort_values(by=["founder_ceased_on", "date_of_creation"], ascending=[False, False], na_position="last")

    stats = {
        "total_matched": total_matched,
        "already_screened": already_screened,
        "newly_screened": len(to_screen),
        "total_screened": len(cache["screened"]),
        "total_hits": len(df) if not df.empty else 0,
    }
    return df, stats


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Companies House Liquidity Screener", layout="wide")
st.title("Companies House Liquidity Event Screener")
st.caption("Screen UK tech and biotech companies (SIC 62012 & 72110) for founder liquidity-event signals.")

with st.sidebar:
    st.header("Configuration")
    _default_key = st.secrets.get("companies_house", {}).get("api_key", "") or os.getenv("CH_API_KEY", "")
    api_key = st.text_input("Companies House API key", type="password", value=_default_key)
    max_companies = st.slider("Max companies to fetch", min_value=50, max_value=1000, value=250, step=50)
    years_back = st.slider("Founded within last N years", min_value=1, max_value=15, value=10)
    sleep_seconds = st.slider("API throttle (seconds per call)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.divider()
    if st.button("🗑️ Clear cache & re-screen all", type="secondary"):
        if Path(CACHE_FILE).exists():
            Path(CACHE_FILE).unlink()
        st.cache_data.clear()
        st.success("Cache cleared. Run the screen to start fresh.")

    run = st.button("▶ Run screen", type="primary")

st.markdown("""
**How it works:** Searches active companies with SIC codes 62012 / 72110, founded within the selected period.
PSC records are checked for 25%+ ownership bands and cessation signals. Filing history is scanned for share transfers and capital changes.
Previously screened companies are **skipped automatically** — only new companies are processed each run.
""")

if run:
    if not api_key:
        st.error("Please enter your Companies House API key in the sidebar.")
    else:
        with st.spinner("Fetching company list from Companies House…"):
            df, stats = run_screen(api_key, max_companies, years_back, sleep_seconds)

        # --- Summary metrics table ---
        st.subheader("Screening summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Companies matched (SIC)", stats["total_matched"])
        col2.metric("Already in cache", stats["already_screened"])
        col3.metric("Newly screened", stats["newly_screened"])
        col4.metric("Total screened to date", stats["total_screened"])
        col5.metric("Total hits", stats["total_hits"])

        st.divider()

        if df.empty:
            st.warning("No candidate liquidity events found yet.")
        else:
            st.success(f"Showing {len(df)} candidate founder liquidity-event rows.")

            # Build display copy with hyperlinks embedded as "Here"
            display_df = df.copy()
            display_df["company_profile_link"] = display_df["company_profile_link"].apply(
                lambda url: f'<a href="{url}" target="_blank">Here</a>' if url else ""
            )
            display_df["liquidity_signal"] = display_df["liquidity_signal"].apply(
                lambda txt: (txt[:120] + "…") if isinstance(txt, str) and len(txt) > 120 else txt
            )

            # Rename columns for display
            display_df = display_df.rename(columns={
                "target_company_name": "Company",
                "company_number": "Co. No.",
                "sic_codes": "SIC",
                "date_of_creation": "Incorporated",
                "founder_name": "Founder",
                "founder_postcode": "Postcode",
                "founder_dob": "DOB (m/y)",
                "founder_control_band": "Control Band",
                "founder_ceased_on": "PSC Ceased",
                "capital_injected": "Capital",
                "purchaser_company_name": "Purchaser",
                "liquidity_signal": "Signal",
                "company_profile_link": "CH Profile",
            })

            st.write(
                display_df.to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download results as CSV",
                data=csv_data,
                file_name="companies_house_liquidity_screen.csv",
                mime="text/csv",
            )
