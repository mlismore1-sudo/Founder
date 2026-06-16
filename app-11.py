import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

API_BASE = "https://api.company-information.service.gov.uk"
TECH_SIC_CODES = ["62012", "72110"]
DEFAULT_YEARS = 10


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

    def get_filing_history(self, company_number: str, items_per_page: int = 100, start_index: int = 0) -> Dict:
        return self._get(
            f"/company/{company_number}/filing-history",
            params={"items_per_page": items_per_page, "start_index": start_index},
        )

    def list_pscs(self, company_number: str) -> Dict:
        return self._get(f"/company/{company_number}/persons-with-significant-control")

    def get_psc_details(self, company_number: str, item: Dict) -> Dict:
        links = item.get("links", {}).get("self")
        if not links:
            return {}
        path = links.replace("/company", "") if links.startswith("/company") else links
        if not path.startswith("/"):
            path = "/" + path
        return self._get(path)


LIQUIDITY_PATTERNS = [
    r"share allotment",
    r"statement of capital",
    r"cessation",
    r"psc",
    r"person with significant control",
    r"transfer",
    r"acquisition",
    r"articles",
    r"confirmation statement",
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
    cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return (cutoff.year - created.year) <= years_back


def psc_is_founder_like(psc: Dict, incorporation_date: Optional[datetime]) -> bool:
    ceased = psc.get("ceased_on")
    notified = parse_date(psc.get("notified_on"))
    if ceased:
        return True
    if incorporation_date and notified:
        return abs((notified - incorporation_date).days) <= 730
    return False


def extract_control_band(natures: List[str]) -> str:
    nature_text = " | ".join(natures or [])
    if "75-to-100" in nature_text:
        return "75-100%"
    if "50-to-75" in nature_text:
        return "50-75%"
    if "25-to-50" in nature_text:
        return "25-50%"
    return "Unknown"


def scan_filings_for_liquidity_signal(filings: List[Dict]) -> Tuple[bool, str, str, str]:
    matched = []
    purchaser = ""
    capital = ""
    for item in filings:
        title = (item.get("description") or "") + " " + (item.get("description_values") or "") if isinstance(item.get("description_values"), str) else (item.get("description") or "")
        desc = f"{item.get('category','')} {item.get('type','')} {title}".lower()
        if any(re.search(pattern, desc) for pattern in LIQUIDITY_PATTERNS):
            matched.append(f"{item.get('date','unknown')}: {item.get('type','')} - {item.get('description','')}")
            if not capital:
                m = re.search(r"([£$€]\s?[\d,]+(?:\.\d+)?)", desc, re.I)
                if m:
                    capital = m.group(1)
            if not purchaser:
                for p in PURCHASER_HINTS:
                    m = re.search(p, item.get("description", ""), re.I)
                    if m:
                        purchaser = m.group(1).strip(" .,")
                        break
    return bool(matched), " ; ".join(matched[:8]), purchaser, capital


def get_all_target_companies(client: CompaniesHouseClient, sic_codes: List[str], max_companies: int) -> List[Dict]:
    companies = []
    start_index = 0
    page_size = 100
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


def build_founder_rows(client: CompaniesHouseClient, company_number: str, incorporation_date: Optional[datetime]) -> List[Dict]:
    psc_list = client.list_pscs(company_number).get("items", [])
    founders = []
    for item in psc_list:
        detail = client.get_psc_details(company_number, item)
        natures = detail.get("natures_of_control") or item.get("natures_of_control") or []
        if not any("ownership-of-shares" in n for n in natures):
            continue
        if not psc_is_founder_like(detail or item, incorporation_date):
            continue
        address = detail.get("address", {}) or {}
        dob = detail.get("date_of_birth", {}) or {}
        name = detail.get("name") or item.get("name") or "Unknown"
        founders.append(
            {
                "founder_name": name,
                "founder_postcode": address.get("postal_code", ""),
                "founder_dob": f"{dob.get('month','')}/{dob.get('year','')}".strip("/"),
                "founder_control_band": extract_control_band(natures),
                "founder_ceased_on": detail.get("ceased_on") or item.get("ceased_on") or "",
                "founder_notified_on": detail.get("notified_on") or item.get("notified_on") or "",
            }
        )
    return founders


def analyse_company(client: CompaniesHouseClient, company_item: Dict, years_back: int) -> List[Dict]:
    company_number = company_item.get("company_number")
    profile = client.get_company_profile(company_number)
    if not company_within_years(profile, years_back):
        return []

    incorporation_date = parse_date(profile.get("date_of_creation"))
    filings = client.get_filing_history(company_number, items_per_page=100).get("items", [])
    has_signal, signal_text, purchaser_name, capital = scan_filings_for_liquidity_signal(filings)
    founders = build_founder_rows(client, company_number, incorporation_date)

    output = []
    for founder in founders:
        if founder["founder_ceased_on"] or has_signal:
            output.append(
                {
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
                    "company_profile_link": f"https://find-and-update.company-information.service.gov.uk/company/{company_number}",
                }
            )
    return output


@st.cache_data(show_spinner=False, ttl=3600)
def run_screen(api_key: str, max_companies: int, years_back: int, sleep_seconds: float) -> pd.DataFrame:
    client = CompaniesHouseClient(api_key=api_key, sleep_seconds=sleep_seconds)
    companies = get_all_target_companies(client, TECH_SIC_CODES, max_companies=max_companies)
    rows = []
    for company in companies:
        try:
            rows.extend(analyse_company(client, company, years_back))
        except requests.HTTPError:
            continue
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["company_number", "founder_name", "founder_ceased_on", "purchaser_company_name"])
        df = df.sort_values(by=["founder_ceased_on", "date_of_creation"], ascending=[False, False], na_position="last")
    return df


st.set_page_config(page_title="Companies House Liquidity Screener", layout="wide")
st.title("Companies House Liquidity Event Screener")
st.caption("Screen UK tech and biotech companies for founder liquidity-event signals using Companies House public data.")

with st.sidebar:
    st.header("Configuration")
    _default_key = st.secrets.get("companies_house", {}).get("api_key", "") or os.getenv("CH_API_KEY", "")
    api_key = st.text_input("Companies House API key", type="password", value=_default_key)
    max_companies = st.slider("Max companies to scan", min_value=50, max_value=1000, value=250, step=50)
    years_back = st.slider("Founded within last N years", min_value=1, max_value=15, value=10)
    sleep_seconds = st.slider("Throttle between API calls (seconds)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    run = st.button("Run screen", type="primary")

st.markdown(
    """
### What this app does
- Searches active companies using SIC codes 62012 and 72110.
- Filters to companies founded within the last selected number of years.
- Pulls PSC records to identify individuals with 25%+ ownership bands.
- Flags potential liquidity events using PSC cessations and filing-history signals.

### Important limitations
- Companies House does **not** provide a clean, explicit API field saying a founder sold shares to an acquirer.
- Purchaser names and capital injected are often unavailable unless inferable from filing descriptions.
- Founder identification is heuristic: this app treats early PSC shareholders as founder-like candidates.
"""
)

if run:
    if not api_key:
        st.error("Please enter your Companies House API key.")
    else:
        with st.spinner("Scanning Companies House data..."):
            df = run_screen(api_key, max_companies, years_back, sleep_seconds)

        if df.empty:
            st.warning("No candidate liquidity events were found with the current filters.")
        else:
            st.success(f"Found {len(df)} candidate founder liquidity-event rows.")
            st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name="companies_house_liquidity_screen.csv",
                mime="text/csv",
            )
            st.subheader("Recommended next step")
            st.write(
                "For the strongest results, enrich these candidates with filing-document downloads, Crunchbase/Dealroom news, LinkedIn founder matching, and acquirer registry checks."
            )
