# ============================================================
# constants.py  –  Country Mappings & Region Codes
# ============================================================
# Central place for all hard-coded categories, mappings, and
# column definitions used throughout the project.
# ============================================================


# SCB birth region categories (Swedish national statistics groupings)
BIRTH_REGION_CATEGORIES = [
    'Sweden',
    'Nordic (excl. Sweden)',
    'EU/EFTA (excl. Nordic)',
    'Europe (excl. EU/EFTA)',
    'Africa',
    'Asia',
    'North America',
    'South America',
    'Oceania',
]


# Map specific countries to standardised birth-region labels
# (used when merging Migrationsverket data with SCB categories)
COUNTRY_TO_REGION = {
    # Nordic
    'Denmark':     'Nordic (excl. Sweden)',
    'Finland':     'Nordic (excl. Sweden)',
    'Norway':      'Nordic (excl. Sweden)',
    'Iceland':     'Nordic (excl. Sweden)',

    # EU / EFTA
    'Germany':     'EU/EFTA (excl. Nordic)',
    'Poland':      'EU/EFTA (excl. Nordic)',
    'France':      'EU/EFTA (excl. Nordic)',
    'Romania':     'EU/EFTA (excl. Nordic)',
    'Italy':       'EU/EFTA (excl. Nordic)',

    # Middle East / Western Asia
    'Syria':       'Asia',
    'Iraq':        'Asia',
    'Iran':        'Asia',
    'Afghanistan': 'Asia',
    'Turkey':      'Asia',

    # Africa
    'Somalia':     'Africa',
    'Eritrea':     'Africa',
    'Ethiopia':    'Africa',
    'Morocco':     'Africa',

    # South Asia
    'India':       'Asia',
    'Pakistan':    'Asia',
    'Bangladesh':  'Asia',

    # East Asia
    'China':       'Asia',
    'Thailand':    'Asia',
    'Vietnam':     'Asia',
    'Philippines': 'Asia',

    # Americas
    'USA':         'North America',
    'Chile':       'South America',
    'Colombia':    'South America',
}


# Integration outcome column definitions
# Keys are the column names used internally; values are display labels.
OUTCOME_COLUMNS = {
    'employment_rate':       'Employment Rate (%)',
    'median_income':         'Median Income (SEK)',
    'self_sufficiency_rate': 'Self-Sufficiency Rate (%)',
    'welfare_recipients_pct': 'Welfare Recipients (%)',
    'welfare_amount_avg':    'Avg Welfare Amount (SEK)',
}
