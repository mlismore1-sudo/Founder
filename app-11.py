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

API_BASE = 'https://api.company-information.service.gov.uk'
TECH_SIC_CODES = ['62012', '72110']
CACHE_FILE = 'screened_companies_cache.json'

CORPORATE_KINDS = (
    'corporate-entity-person-with-significant-control',
    'corporate-entity-beneficial-owner',
    'legal-person-with-significant-control',
    'legal-person-beneficial-owner',
)

INDIVIDUAL_KINDS = (
    'individual-person-with-significant-control',
    'individual-beneficial-owner',
)

SHARE_OWNERSHIP_NATURES = (
    'ownership-of-shares-25-to-50-percent',
    'ownership-of-shares-50-to-75-percent',
    'ownership-of-shares-75-to-100-percent',
    'ownership-of-shares-25-to-50-percent-as-trust',
    'ownership-of-shares-50-to-75-percent-as-trust',
    'ownership-of-shares-75-to-100-percent-as-trust',
    'ownership-of-shares-25-to-50-percent-as-firm',
    'ownership-of-shares-50-to-75-percent-as-firm',
    'ownership-of-shares-75-to-100-percent-as-firm',
    'ownership-of-shares-more-than-25-percent-registered-overseas-entity',
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> Dict:
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {'screened': {}, 'hits': [], 'next_start_index': 0}


def save_cache(cache: Dict) -> None:
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def mark_screened(cache: Dict, company_number: str, rows: List[Dict]) -> None:
    cache['screened'][company_number] = True
    if rows:
        existing = {(r['company_number'], r['founder_name'], r['acquirer_company_name']) for r in cache['hits']}
        for row in rows:
            key = (row['company_number'], row['founder_name'], row['acquirer_company_name'])
            if key not in existing:
                cache['hits'].append(row)
                existing.add(key)


# ---------------------------------------------------------------------------
# Companies House API client
# ---------------------------------------------------------------------------

class CompaniesHouseClient:
    def __init__(self, api_key: str, sleep_seconds: float = 0.2):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.auth = (api_key, '')
        self.session.headers.update({'User-Agent': 'streamlit-liquidity-event-screener/1.0'})

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f'{API_BASE}{path}'
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        time.sleep(self.sleep_seconds)
        return resp.json()

    def advanced_company_search(self, sic_codes: List[str], size: int = 100, start_index: int = 0) -> Dict:
        return self._get('/advanced-search/companies', params={
            'size': size,
            'start_index': start_index,
            'sic_codes': ','.join(sic_codes),
            'company_status': 'active',
        })

    def get_company_profile(self, company_number: str) -> Dict:
        return self._get(f'/company/{company_number}')

    def get_filing_history(self, company_number: str, items_per_page: int = 100) -> Dict:
        return self._get(f'/company/{company_number}/filing-history',
                         params={'items_per_page': items_per_page})

    def list_pscs(self, company_number: str) -> List[Dict]:
        return self._get(f'/company/{company_number}/persons-with-significant-control').get('items', [])


# ---------------------------------------------------------------------------
# PSC helpers
# ---------------------------------------------------------------------------

def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def has_share_ownership_25plus(psc: Dict) -> bool:
    natures = psc.get('natures_of_control') or []
    return any(n in SHARE_OWNERSHIP_NATURES for n in natures)


def extract_control_band(natures: List[str]) -> str:
    text = ' | '.join(natures or [])
    if '75-to-100' in text: return '75-100%'
    if '50-to-75' in text:  return '50-75%'
    if '25-to-50' in text:  return '25-50%'
    if 'more-than-25' in text: return '>25%'
    return 'Unknown'


def is_individual_psc(psc: Dict) -> bool:
    kind = psc.get('kind') or ''
    return any(kind.startswith(k) for k in INDIVIDUAL_KINDS)


def is_corporate_psc(psc: Dict) -> bool:
    kind = psc.get('kind') or ''
    return any(kind.startswith(k) for k in CORPORATE_KINDS)


def psc_is_founder_like(psc: Dict, incorporation_date: Optional[datetime]) -> bool:
    if psc.get('ceased_on'):
        return True
    notified = parse_date(psc.get('notified_on'))
    if incorporation_date and notified:
        return abs((notified - incorporation_date).days) <= 730
    return False


def find_corporate_acquirers(pscs: List[Dict], incorporation_date: Optional[datetime]) -> List[Dict]:
    acquirers = []
    for psc in pscs:
        if not is_corporate_psc(psc):
            continue
        if not has_share_ownership_25plus(psc):
            continue
        notified = parse_date(psc.get('notified_on'))
        if incorporation_date and notified:
            if (notified - incorporation_date).days < 30:
                continue
        acquirers.append(psc)
    return acquirers


def find_founder_individuals(pscs: List[Dict], incorporation_date: Optional[datetime]) -> List[Dict]:
    return [
        psc for psc in pscs
        if is_individual_psc(psc)
        and has_share_ownership_25plus(psc)
        and psc_is_founder_like(psc, incorporation_date)
    ]


def extract_capital_from_filings(filings: List[Dict]) -> str:
    for item in filings:
        desc = (item.get('description') or '').lower()
        ftype = (item.get('type') or '').lower()
        if 'statement of capital' in desc or 'share allotment' in desc or 'sh01' in ftype:
            m = re.search(r'([£$€]\s?[\d,]+(?:\.\d+)?)', desc, re.I)
            if m:
                return m.group(1)
    return ''


# ---------------------------------------------------------------------------
# Analyse a single company
# ---------------------------------------------------------------------------

def is_active(profile: Dict) -> bool:
    return (profile.get('company_status') or '').lower() == 'active'


def company_within_years(profile: Dict, years_back: int) -> bool:
    created = parse_date(profile.get('date_of_creation'))
    if not created:
        return False
    return (datetime.now().year - created.year) <= years_back


def analyse_company(client: CompaniesHouseClient, company_item: Dict, years_back: int) -> List[Dict]:
    company_number = company_item.get('company_number')
    profile = client.get_company_profile(company_number)

    if not is_active(profile):
        return []
    if not company_within_years(profile, years_back):
        return []

    incorporation_date = parse_date(profile.get('date_of_creation'))
    pscs = client.list_pscs(company_number)

    acquirers = find_corporate_acquirers(pscs, incorporation_date)
    if not acquirers:
        return []

    founders = find_founder_individuals(pscs, incorporation_date)
    if not founders:
        return []

    filings = client.get_filing_history(company_number).get('items', [])
    capital = extract_capital_from_filings(filings)
    ch_url = f'https://find-and-update.company-information.service.gov.uk/company/{company_number}'

    rows = []
    for acquirer in acquirers:
        ident = acquirer.get('identification') or {}
        acquirer_name = acquirer.get('name') or ident.get('legal_name') or 'Unknown'
        for founder in founders:
            address = founder.get('address') or {}
            dob = founder.get('date_of_birth') or {}
            dob_str = '/'.join(filter(None, [str(dob.get('month', '')), str(dob.get('year', ''))]))
            rows.append({
                'target_company_name':   profile.get('company_name'),
                'company_number':        company_number,
                'sic_codes':             ', '.join(profile.get('sic_codes') or []),
                'date_of_creation':      profile.get('date_of_creation') or '',
                'founder_name':          founder.get('name') or 'Unknown',
                'founder_postcode':      address.get('postal_code') or '',
                'founder_dob':           dob_str,
                'founder_control_band':  extract_control_band(founder.get('natures_of_control') or []),
                'founder_ceased_on':     founder.get('ceased_on') or '',
                'acquirer_company_name': acquirer_name,
                'acquirer_notified_on':  acquirer.get('notified_on') or '',
                'acquirer_ceased_on':    acquirer.get('ceased_on') or '',
                'acquirer_control_band': extract_control_band(acquirer.get('natures_of_control') or []),
                'acquirer_country':      ident.get('country_registered') or (acquirer.get('address') or {}).get('country') or '',
                'capital_injected':      capital,
                'company_profile_link':  ch_url,
            })
    return rows


# ---------------------------------------------------------------------------
# Main screening loop
# ---------------------------------------------------------------------------

def get_next_companies(client: CompaniesHouseClient, sic_codes: List[str],
                       batch_size: int, start_index: int) -> Tuple[List[Dict], int]:
    companies = []
    current_index = start_index
    page_size = 100
    while len(companies) < batch_size:
        fetch_size = min(page_size, batch_size - len(companies))
        data = client.advanced_company_search(sic_codes=sic_codes, size=fetch_size,
                                              start_index=current_index)
        items = data.get('items') or []
        if not items:
            break
        companies.extend(items)
        current_index += len(items)
        if len(items) < fetch_size:
            break
    return companies, current_index


def run_screen(api_key: str, batch_size: int, years_back: int, sleep_seconds: float) -> Tuple[pd.DataFrame, Dict]:
    client = CompaniesHouseClient(api_key=api_key, sleep_seconds=sleep_seconds)
    cache = load_cache()

    start_index = cache.get('next_start_index', 0)
    companies, new_start_index = get_next_companies(client, TECH_SIC_CODES, batch_size, start_index)

    total_fetched = len(companies)
    to_screen = [c for c in companies if c.get('company_number') not in cache['screened']]
    skipped = total_fetched - len(to_screen)

    if to_screen:
        progress_bar = st.progress(0, text='Starting scan...')
        for i, company in enumerate(to_screen):
            try:
                rows = analyse_company(client, company, years_back)
                mark_screened(cache, company['company_number'], rows)
            except Exception:
                mark_screened(cache, company['company_number'], [])
            pct = int((i + 1) / len(to_screen) * 100)
            progress_bar.progress(pct, text=f'Screened {i + 1} of {len(to_screen)} companies...')
        progress_bar.empty()

    cache['next_start_index'] = new_start_index
    save_cache(cache)

    if not to_screen:
        st.info(
            f'All {total_fetched} companies in this batch were already cached. '
            f'The next run will begin at index {new_start_index}.'
        )

    all_hits = cache.get('hits') or []
    df = pd.DataFrame(all_hits) if all_hits else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=['company_number', 'founder_name', 'acquirer_company_name'])
        df = df.sort_values(by=['acquirer_notified_on', 'date_of_creation'],
                            ascending=[False, False], na_position='last')

    stats = {
        'batch_fetched':    total_fetched,
        'skipped_cached':   skipped,
        'newly_screened':   len(to_screen),
        'total_screened':   len(cache['screened']),
        'next_start_index': cache['next_start_index'],
        'total_hits':       len(df) if not df.empty else 0,
    }
    return df, stats


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title='CH Liquidity Event Screener', layout='wide')
st.title('Companies House Liquidity Event Screener')
st.caption(
    'Identifies UK tech & biotech companies (SIC 62012 / 72110) where a corporate entity '
    'has acquired 25%+ of shares from a founder post-incorporation.'
)

with st.sidebar:
    st.header('Configuration')
    _default_key = (st.secrets.get('companies_house') or {}).get('api_key', '') or os.getenv('CH_API_KEY', '')
    api_key = st.text_input('Companies House API key', type='password', value=_default_key)
    batch_size = st.slider('Companies to screen this run', min_value=50, max_value=1000, value=250, step=50)
    years_back = st.slider('Founded within last N years', min_value=1, max_value=15, value=10)
    sleep_seconds = st.slider('API throttle (secs per call)', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    _cache = load_cache()
    st.divider()
    st.caption(f'Next run starts at index: **{_cache.get("next_start_index", 0)}**')
    st.caption(f'Total screened to date: **{len(_cache.get("screened", {}))}**')
    st.divider()
    if st.button('Clear cache and restart from index 0', type='secondary'):
        if Path(CACHE_FILE).exists():
            Path(CACHE_FILE).unlink()
        st.rerun()
    run = st.button('Run next batch', type='primary')

with st.expander('How signals are detected', expanded=False):
    st.markdown(
        '**A match requires ALL of the following:**\n\n'
        '1. Company is **active** and incorporated within the selected window.\n'
        '2. At least one **individual PSC** holds 25%+ of shares and was notified within 2 years of incorporation, or has since ceased.\n'
        '3. At least one **corporate-entity PSC** holds 25%+ of shares AND was notified **more than 30 days after incorporation**.\n\n'
        '**Pagination:** Each run fetches the next unseen batch. The cursor advances automatically so no company is double-screened.'
    )

if run:
    if not api_key:
        st.error('Please enter your Companies House API key in the sidebar.')
    else:
        df, stats = run_screen(api_key, batch_size, years_back, sleep_seconds)

        st.subheader('Screening summary')
        cols = st.columns(6)
        cols[0].metric('Fetched this run',      stats['batch_fetched'])
        cols[1].metric('Skipped (cached)',       stats['skipped_cached'])
        cols[2].metric('Newly screened',         stats['newly_screened'])
        cols[3].metric('Total screened',         stats['total_screened'])
        cols[4].metric('Next start index',       stats['next_start_index'])
        cols[5].metric('Total hits',             stats['total_hits'])

        st.divider()

        if df.empty:
            st.warning('No corporate acquisition signals found yet.')
        else:
            st.success(f"{len(df)} founder/acquirer rows across {df['company_number'].nunique()} companies.")

            display_df = df.copy()
            display_df['company_profile_link'] = display_df['company_profile_link'].apply(
                lambda u: f'<a href="{u}" target="_blank">Here</a>' if u else ''
            )
            display_df = display_df.rename(columns={
                'target_company_name':   'Company',
                'company_number':        'Co. No.',
                'sic_codes':             'SIC',
                'date_of_creation':      'Incorporated',
                'founder_name':          'Founder',
                'founder_postcode':      'Postcode',
                'founder_dob':           'DOB (m/y)',
                'founder_control_band':  'Founder Band',
                'founder_ceased_on':     'Founder Ceased',
                'acquirer_company_name': 'Acquirer',
                'acquirer_notified_on':  'Acquirer Notified',
                'acquirer_ceased_on':    'Acquirer Ceased',
                'acquirer_control_band': 'Acquirer Band',
                'acquirer_country':      'Acquirer Country',
                'capital_injected':      'Capital',
                'company_profile_link':  'CH Profile',
            })

            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download all results as CSV',
                data=csv_data,
                file_name='ch_liquidity_screen.csv',
                mime='text/csv',
            )
