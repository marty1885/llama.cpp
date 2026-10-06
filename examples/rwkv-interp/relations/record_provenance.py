"""Record local model, source, binary, and platform identity for a collected run."""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=Path)
parser.add_argument('model', type=Path)
parser.add_argument('collector', type=Path)
parser.add_argument('source', type=Path)
parser.add_argument('repository', type=Path)
args = parser.parse_args()

def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()

repo = args.repository.resolve()
model = args.model.resolve()
files = [args.collector, args.source, args.source.parent/'rwkv-session.h', args.dataset/'plan.tsv', args.dataset/'prompts.jsonl']
manifest = dict(
    format_version=1,
    recorded_at=datetime.now(timezone.utc).isoformat(),
    platform=platform.platform(),
    machine=platform.machine(),
    python=platform.python_version(),
    model=dict(path=str(model), bytes=model.stat().st_size, sha256=digest(model)),
    git_head=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    git_status=subprocess.check_output(['git', '-C', str(repo), 'status', '--short'], text=True),
    files=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p)) for p in files],
    execution=dict(backend='CPU', state_dtype='F32', threads=8, batch=1, ubatch=1, sequences=1, rollback=0, bos=False, chat_template=False),
)
(args.dataset/'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Recorded model and collection provenance')
