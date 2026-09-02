# WRDS schema notes

**Verified on 2026-09-02** by `scripts/00_verify_schema.py` against the live WRDS server.
This file is generated -- re-run the script rather than editing it by hand.
Row counts are the query planner's estimate, not an exact `COUNT(*)`.
Stage 0 of `docs/PHASE1_PLAN.md`; every later stage reads table names from
`docs/wrds_schema_resolved.json`, never from a hard-coded guess.

## Resolved tables

| Need | Plan guess | Resolved | Rows (est.) | Date range | Status |
|---|---|---|---:|---|---|
| S&P 500 point-in-time membership | `crsp.dsp500list` | `crsp.dsp500list` | 2,064 | — | ✅ |
| Monthly market cap | `crsp.msf` | `crsp.msf` | 5,153,763 | 1925-12-31 → 2024-12-31 | ✅ |
| Daily stock prices | `crsp.dsf` | `crsp.dsf` | 107,682,104 | 1925-12-31 → 2024-12-31 | ✅ |
| CRSP name history | `crsp.dsenames` | `crsp.dsenames` | 117,859 | — | ✅ |
| Delisting returns | `crsp.dsedelist` | `crsp.dsedelist` | 38,872 | — | ✅ |
| Daily option prices (2019 probe) | `optionm.opprcd2019` | `optionm.opprcd2019` | 233,948,704 | 2019-01-02 → 2019-12-31 | ✅ |
| OptionMetrics security master | `optionm.securd` | `optionm.securd` | 116,178 | — | ✅ |
| Zero-coupon curve | `optionm.zerocd` | `optionm.zerocd` | 304,301 | 1996-01-02 → 2025-08-29 | ✅ |
| Projected dividends (discrete) | `optionm.distrd` | `optionm.distrd` | 743,101 | 1900-01-01 → 2031-08-04 | ✅ |
| Projected dividends (point-in-time projection) | `optionm.distrprojd2019` | `optionm.distrprojd2019` | 11,877,294 | — | ✅ |
| OM historical volatility | `optionm.hvold` | `optionm.hvold2019` | 13,910,557 | 2019-01-02 → 2019-12-31 | ✅ via fallback |
| OM <-> CRSP link | `wrdsapps.opcrsphist` | `wrdsapps.opcrsphist` | 121,773 | — | ✅ |

## Column lists

### `crsp.dsp500list` — S&P 500 point-in-time membership

| column | type |
|---|---|
| `permno` | INTEGER |
| `start` | DATE |
| `ending` | DATE |

### `crsp.msf` — Monthly market cap

| column | type |
|---|---|
| `cusip` | VARCHAR(8) |
| `permno` | INTEGER |
| `permco` | INTEGER |
| `issuno` | INTEGER |
| `hexcd` | SMALLINT |
| `hsiccd` | INTEGER |
| `date` | DATE |
| `bidlo` | NUMERIC(11, 5) |
| `askhi` | NUMERIC(11, 5) |
| `prc` | NUMERIC(11, 5) |
| `vol` | NUMERIC(10, 0) |
| `ret` | NUMERIC(10, 6) |
| `bid` | NUMERIC(11, 5) |
| `ask` | NUMERIC(11, 5) |
| `shrout` | DOUBLE PRECISION |
| `cfacpr` | DOUBLE PRECISION |
| `cfacshr` | DOUBLE PRECISION |
| `altprc` | NUMERIC(11, 5) |
| `spread` | NUMERIC(10, 5) |
| `altprcdt` | DATE |
| `retx` | NUMERIC(10, 6) |

### `crsp.dsf` — Daily stock prices

| column | type |
|---|---|
| `cusip` | VARCHAR(8) |
| `permno` | INTEGER |
| `permco` | INTEGER |
| `issuno` | INTEGER |
| `hexcd` | SMALLINT |
| `hsiccd` | INTEGER |
| `date` | DATE |
| `bidlo` | NUMERIC(11, 5) |
| `askhi` | NUMERIC(11, 5) |
| `prc` | NUMERIC(11, 5) |
| `vol` | NUMERIC(10, 0) |
| `ret` | NUMERIC(10, 6) |
| `bid` | NUMERIC(11, 5) |
| `ask` | NUMERIC(11, 5) |
| `shrout` | DOUBLE PRECISION |
| `cfacpr` | DOUBLE PRECISION |
| `cfacshr` | DOUBLE PRECISION |
| `openprc` | NUMERIC(11, 5) |
| `numtrd` | INTEGER |
| `retx` | NUMERIC(10, 6) |

### `crsp.dsenames` — CRSP name history

| column | type |
|---|---|
| `permno` | INTEGER |
| `namedt` | DATE |
| `nameendt` | DATE |
| `shrcd` | SMALLINT |
| `exchcd` | SMALLINT |
| `siccd` | INTEGER |
| `ncusip` | VARCHAR(8) |
| `ticker` | VARCHAR(8) |
| `comnam` | VARCHAR(35) |
| `shrcls` | VARCHAR(4) |
| `tsymbol` | VARCHAR(10) |
| `naics` | VARCHAR(7) |
| `primexch` | VARCHAR(1) |
| `trdstat` | VARCHAR(1) |
| `secstat` | VARCHAR(1) |
| `permco` | INTEGER |
| `compno` | INTEGER |
| `issuno` | INTEGER |
| `hexcd` | SMALLINT |
| `hsiccd` | INTEGER |
| `cusip` | VARCHAR(8) |

### `crsp.dsedelist` — Delisting returns

| column | type |
|---|---|
| `permno` | INTEGER |
| `dlstdt` | DATE |
| `dlstcd` | SMALLINT |
| `nwperm` | INTEGER |
| `nwcomp` | INTEGER |
| `nextdt` | DATE |
| `dlamt` | NUMERIC(11, 5) |
| `dlretx` | NUMERIC(10, 6) |
| `dlprc` | NUMERIC(11, 5) |
| `dlpdt` | DATE |
| `dlret` | NUMERIC(10, 6) |
| `permco` | INTEGER |
| `compno` | INTEGER |
| `issuno` | INTEGER |
| `hexcd` | SMALLINT |
| `hsiccd` | INTEGER |
| `cusip` | VARCHAR(8) |
| `acperm` | DOUBLE PRECISION |
| `accomp` | DOUBLE PRECISION |

### `optionm.opprcd2019` — Daily option prices (2019 probe)

| column | type |
|---|---|
| `secid` | DOUBLE PRECISION |
| `date` | DATE |
| `symbol` | VARCHAR(21) |
| `symbol_flag` | VARCHAR(1) |
| `exdate` | DATE |
| `last_date` | DATE |
| `cp_flag` | VARCHAR(1) |
| `strike_price` | DOUBLE PRECISION |
| `best_bid` | DOUBLE PRECISION |
| `best_offer` | DOUBLE PRECISION |
| `volume` | DOUBLE PRECISION |
| `open_interest` | DOUBLE PRECISION |
| `impl_volatility` | DOUBLE PRECISION |
| `delta` | DOUBLE PRECISION |
| `gamma` | DOUBLE PRECISION |
| `vega` | DOUBLE PRECISION |
| `theta` | DOUBLE PRECISION |
| `optionid` | DOUBLE PRECISION |
| `cfadj` | DOUBLE PRECISION |
| `am_settlement` | DOUBLE PRECISION |
| `contract_size` | DOUBLE PRECISION |
| `ss_flag` | VARCHAR(1) |
| `forward_price` | DOUBLE PRECISION |
| `expiry_indicator` | VARCHAR(1) |
| `root` | VARCHAR(5) |
| `suffix` | VARCHAR(2) |

### `optionm.securd` — OptionMetrics security master

| column | type |
|---|---|
| `secid` | DOUBLE PRECISION |
| `cusip` | VARCHAR(8) |
| `ticker` | VARCHAR(6) |
| `sic` | VARCHAR(4) |
| `index_flag` | VARCHAR(1) |
| `exchange_d` | DOUBLE PRECISION |
| `class` | VARCHAR(1) |
| `issue_type` | VARCHAR(1) |
| `industry_group` | DOUBLE PRECISION |

### `optionm.zerocd` — Zero-coupon curve

| column | type |
|---|---|
| `date` | DATE |
| `days` | DOUBLE PRECISION |
| `rate` | DOUBLE PRECISION |

### `optionm.distrd` — Projected dividends (discrete)

| column | type |
|---|---|
| `secid` | DOUBLE PRECISION |
| `record_date` | DATE |
| `seq_num` | DOUBLE PRECISION |
| `ex_date` | DATE |
| `amount` | DOUBLE PRECISION |
| `adj_factor` | DOUBLE PRECISION |
| `declare_date` | DATE |
| `payment_date` | DATE |
| `link_secid` | DOUBLE PRECISION |
| `distr_type` | VARCHAR(1) |
| `frequency` | VARCHAR(1) |
| `currency` | VARCHAR(3) |
| `approx_flag` | VARCHAR(1) |
| `cancel_flag` | VARCHAR(1) |
| `liquid_flag` | VARCHAR(1) |

### `optionm.distrprojd2019` — Projected dividends (point-in-time projection)

| column | type |
|---|---|
| `secid` | DOUBLE PRECISION |
| `date` | DATE |
| `exdate` | DATE |
| `amount` | DOUBLE PRECISION |

### `optionm.hvold2019` — OM historical volatility

_resolved via fallback (guess `optionm.hvold` does not exist)_

| column | type |
|---|---|
| `secid` | DOUBLE PRECISION |
| `date` | DATE |
| `days` | DOUBLE PRECISION |
| `volatility` | DOUBLE PRECISION |

### `wrdsapps.opcrsphist` — OM <-> CRSP link

| column | type |
|---|---|
| `secid` | DOUBLE PRECISION |
| `sdate` | DATE |
| `edate` | DATE |
| `permno` | INTEGER |
| `score` | DOUBLE PRECISION |

## Checkpoint 0 probes

### link_score_distribution

```json
[
  {
    "score": 1.0,
    "n": 28336,
    "n_permno": 27997,
    "n_secid": 27887
  },
  {
    "score": 2.0,
    "n": 190,
    "n_permno": 181,
    "n_secid": 189
  },
  {
    "score": 3.0,
    "n": 8,
    "n_permno": 8,
    "n_secid": 8
  },
  {
    "score": 4.0,
    "n": 660,
    "n_permno": 404,
    "n_secid": 653
  },
  {
    "score": 5.0,
    "n": 5687,
    "n_permno": 3382,
    "n_secid": 5190
  },
  {
    "score": 6.0,
    "n": 86892,
    "n_permno": 0,
    "n_secid": 86892
  }
]
```

### aapl_options_probe

```json
{
  "probe_date": "2019-01-02",
  "ticker": "AAPL",
  "permno": 14593,
  "link_rows": 1,
  "secid": 101594,
  "absent_columns": [
    "exercise_style"
  ],
  "option_rows": 1590,
  "strike_range": [
    2.5,
    425.0
  ],
  "n_expiries": 14,
  "ss_flag_dtype": "string",
  "ss_flag_values": [
    "0"
  ],
  "contract_size_dtype": "Float64",
  "contract_size_values": [
    "100.0"
  ],
  "am_settlement_dtype": "Float64",
  "am_settlement_values": [
    "0.0"
  ],
  "expiry_indicator_dtype": "string",
  "expiry_indicator_values": [
    "w"
  ],
  "nearest_atm": {
    "cp_flag": "P",
    "strike": 175.0,
    "delta": -0.500258,
    "best_bid": 33.2,
    "best_offer": 36.2,
    "impl_volatility": 0.293285
  }
}
```

### dividend_question_q5

```json
{
  "yield_like_tables": [
    "distrd",
    "distribution",
    "distribution_projection",
    "distrprojd1996",
    "distrprojd1997",
    "distrprojd1998",
    "distrprojd1999",
    "distrprojd2000",
    "distrprojd2001",
    "distrprojd2002",
    "distrprojd2003",
    "distrprojd2004",
    "distrprojd2005",
    "distrprojd2006",
    "distrprojd2007",
    "distrprojd2008",
    "distrprojd2009",
    "distrprojd2010",
    "distrprojd2011",
    "distrprojd2012",
    "distrprojd2013",
    "distrprojd2014",
    "distrprojd2015",
    "distrprojd2016",
    "distrprojd2017",
    "distrprojd2018",
    "distrprojd2019",
    "distrprojd2020",
    "distrprojd2021",
    "distrprojd2022",
    "distrprojd2023",
    "idxdvd",
    "index_dividend"
  ],
  "distr_type_counts": [
    {
      "distr_type": "1",
      "n": 174848
    },
    {
      "distr_type": "5",
      "n": 2615
    },
    {
      "distr_type": "2",
      "n": 2573
    },
    {
      "distr_type": "0",
      "n": 511
    },
    {
      "distr_type": "3",
      "n": 395
    },
    {
      "distr_type": "4",
      "n": 327
    },
    {
      "distr_type": "6",
      "n": 141
    },
    {
      "distr_type": "8",
      "n": 42
    },
    {
      "distr_type": "7",
      "n": 7
    },
    {
      "distr_type": "9",
      "n": 3
    }
  ],
  "conclusion": "Build q from discrete projected dividends (escrowed-dividend adjustment, M9/Q5) unless a per-equity continuous yield table appears above."
}
```

