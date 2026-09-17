# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Rebuildable session index, with no new durable state or write protocol.

Each lookup checks inode, size, modification time, and change time. Only changed
case files are parsed. Store locking supplies a consistent cooperating-writer
boundary; the index is never evidence authority and never bypasses replay.
"""
from copy import deepcopy
import logging

log = logging.getLogger(__name__)

class CaseIndex:
    def __init__(self):
        self.entries = {}
        self.by_session = {}
        self.errors = {}

    def refresh(self, store):
        from .store import StoreError, _CASE_ID_RE
        paths = {p.parent.name:p for p in store.cases_dir.glob('*/case.md') if _CASE_ID_RE.fullmatch(p.parent.name)}
        changed = set(paths) != set(self.entries)
        for cid in list(self.entries):
            if cid not in paths:
                del self.entries[cid]
        for cid,path in sorted(paths.items()):
            previous = self.entries.get(cid, {})
            try:
                st = path.stat()
                signature = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            except OSError:
                signature = None
            if signature is not None and previous.get('signature') == signature:
                continue
            changed = True
            try:
                case = store._read_case(path, expected_case_id=cid)
                if case.get('case_id') != cid:
                    raise StoreError('case identity does not match its directory')
                self.entries[cid] = {'signature':signature, 'case':case}
            except (StoreError, OSError) as exc:
                owners = set(previous.get('owners', []))
                old_case = previous.get('case') or {}
                if old_case.get('session_id'):
                    owners.add(old_case['session_id'])
                # After restart there is no cached owner. Timeline attribution
                # conservatively protects every identifiable affected session.
                try:
                    owners.update(str(e['session_id']) for e in store._read_events(path.parent/'events.jsonl') if e.get('session_id'))
                except StoreError:
                    pass
                self.entries[cid] = {'signature':signature,'owners':sorted(owners),'error':type(exc).__name__}
                log.warning('Epistemic case %s is unreadable; unrelated cases remain available',cid)
        if changed:
            self.by_session = {}
            self.errors = {}
            for cid,entry in self.entries.items():
                if 'error' in entry:
                    self.errors[cid] = {'error':entry['error'],'session_ids':entry['owners']}
                else:
                    session = str(entry['case'].get('session_id') or '')
                    self.by_session.setdefault(session, []).append(cid)

    def select(self, store, session_id=None):
        from .store import StoreError
        self.refresh(store)
        if session_id is not None:
            damaged = [cid for cid,entry in self.errors.items() if session_id in entry['session_ids']]
            if damaged:
                raise StoreError('unreadable case associated with this session: '+', '.join(sorted(damaged)))
            ids = self.by_session.get(session_id, [])
        else:
            ids = [cid for cid,entry in self.entries.items() if 'case' in entry]
        return [deepcopy(self.entries[cid]['case']) for cid in sorted(ids)]
