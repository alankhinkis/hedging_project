"""Stage 0's reporting layer, exercised offline.

`render_notes` and `checkpoint_0` are what a reader (and the next stage) actually consume,
so they are tested on synthetic findings rather than being first exercised against a live
WRDS session where a formatting bug would be indistinguishable from a schema problem.
"""

from __future__ import annotations

from vrp.schema import TableFinding, checkpoint_0, render_notes


def finding(**kw) -> TableFinding:
    base = dict(
        need="Daily stock prices",
        guess="crsp.dsf",
        confidence="high",
        resolved="crsp.dsf",
        columns=["permno", "date", "prc"],
        dtypes={"permno": "DOUBLE", "date": "DATE", "prc": "DOUBLE"},
        rows=100_000_000,
        date_min="1925-12-31",
        date_max="2024-12-31",
    )
    base.update(kw)
    return TableFinding(**base)


GOOD_PROBES = {
    "link_score_distribution": [{"score": 1, "n": 10}],
    "aapl_options_probe": {"probe_date": "2019-01-02", "option_rows": 812, "secid": 101594},
    "dividend_question_q5": {"yield_like_tables": ["distrd", "idxdvd"]},
}


def test_checkpoint_0_passes_when_everything_resolves():
    results = checkpoint_0([finding()], GOOD_PROBES)
    assert all(ok for _, ok, _ in results)
    assert len(results) == 5


def test_unresolved_table_fails_the_checkpoint():
    results = dict((n, ok) for n, ok, _ in checkpoint_0([finding(resolved=None)], GOOD_PROBES))
    key = next(k for k in results if k.startswith("every schema target"))
    assert results[key] is False


def test_missing_key_column_fails_the_checkpoint():
    f = finding(missing_key_columns=["cfacpr"])
    results = dict((n, ok) for n, ok, _ in checkpoint_0([f], GOOD_PROBES))
    key = next(k for k in results if k.startswith("every key column"))
    assert results[key] is False


def test_empty_sanity_query_fails_the_checkpoint():
    probes = dict(GOOD_PROBES, aapl_options_probe={"probe_date": "2019-01-02", "option_rows": 0})
    results = dict((n, ok) for n, ok, _ in checkpoint_0([finding()], probes))
    key = next(k for k in results if k.startswith("sanity query"))
    assert results[key] is False


def test_notes_render_includes_fallbacks_and_probes(tmp_path):
    findings = [
        finding(),
        finding(
            need="OM <-> CRSP link",
            guess="wrdsapps.opcrsphist",
            resolved="wrdsapps_link_crsp_optionm.opcrsphist",
            note="resolved via fallback (guess `wrdsapps.opcrsphist` does not exist)",
            columns=["secid", "permno", "score"],
            dtypes={},
        ),
        finding(need="OM historical volatility", guess="optionm.hvold", resolved=None,
                columns=[], dtypes={}, rows=None, date_min=None, date_max=None),
    ]
    md = render_notes(findings, GOOD_PROBES)
    assert "via fallback" in md
    assert "wrdsapps_link_crsp_optionm.opcrsphist" in md
    assert "not found" in md
    assert "aapl_options_probe" in md
    # An unresolved table must not get a column section pretending it exists.
    assert "### `None`" not in md
