# AI-assisted contribution; maintained by Epistemic Harness contributors.
from pathlib import Path
import pytest
from epistemic_harness.harness import EpistemicHarness
from epistemic_harness.store import StoreError

def test_corrupt_unrelated_case_does_not_disable_valid_session(tmp_path):
    h=EpistemicHarness(tmp_path)
    h.model({'operation':'open','case_id':'z-valid','top_unknown':'inspect'},session_id='owner')
    bad=tmp_path/'cases/000-bad';bad.mkdir();(bad/'case.md').write_text('broken fixture')
    assert h.store.get_active_case('owner')['case_id']=='z-valid'
    assert h.store.prepare_injection('unrelated') is None
    assert h.model({'operation':'list'},session_id='other')['cases']
    (tmp_path/'cases/z-valid/case.md').write_text('broken own fixture')
    with pytest.raises(StoreError):h.store.get_active_case('owner')
    # The attribution survives a fresh process-local index via the Timeline.
    with pytest.raises(StoreError):EpistemicHarness(tmp_path).store.get_active_case('owner')

def test_warm_dormant_lookup_does_not_reparse_portfolio(tmp_path,monkeypatch):
    h=EpistemicHarness(tmp_path)
    for n in range(5):h.model({'operation':'open','case_id':f'case-{n}','top_unknown':'inspect'},session_id=f'owner-{n}')
    h.store.get_active_case('unbound');reads=[];old=h.store._read_case
    def read(p, **kwargs):reads.append(p);return old(p, **kwargs)
    monkeypatch.setattr(h.store,'_read_case',read)
    assert h.store.get_active_case('unbound') is None
    assert reads==[]
    other=EpistemicHarness(tmp_path)
    other.model({'operation':'pause','reason':'cross-instance refresh test'},session_id='owner-0')
    assert h.store.get_active_case('owner-0') is None
    assert len(reads)<=1
