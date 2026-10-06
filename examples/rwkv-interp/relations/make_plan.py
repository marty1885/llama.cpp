import argparse
import itertools
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
root = parser.parse_args().output
root.mkdir(parents=True, exist_ok=True)
rows = []
metadata = []

def add(identifier, parent, kind, text, colors, **meta):
    assert '\t' not in text and '\n' not in text
    rows.append('\t'.join((identifier, parent, kind, text.replace('\\n', '\n'), ','.join(colors))))
    # TSV stores escaped newlines, decoded by the collector.
    rows[-1] = rows[-1].replace('\n', '\\n')
    metadata.append(dict(id=identifier, parent=parent, kind=kind, text=text.replace('\\n', '\n'), **meta))

cases = list(itertools.product(('repeated', 'distinct'), (('Japan', 'Canada'), ('France', 'Brazil')), (('yellow', 'blue'), ('green', 'purple')), (0, 1), (0, 1)))
# Same-attribute controls, balanced over identity, location, and order.
for identity, countries, order, shared in itertools.product(('repeated', 'distinct'), (('Japan', 'Canada'), ('France', 'Brazil')), (0, 1), ('yellow', 'blue')):
    cases.append((identity, countries, (shared, shared), order, 0))

for index, (identity, countries, colors, order, assignment) in enumerate(cases):
    names = ('Alice', 'Alice' if identity == 'repeated' else 'Bob')
    assigned = colors if assignment == 0 else colors[::-1]
    entities = [f'{names[i]} in {countries[i]}' for i in range(2)]
    clauses = [f'{entities[i]} has a {assigned[i]} hat.' for i in range(2)]
    story = ' '.join(clauses[::1 if order == 0 else -1])
    base = f'h{index:03d}'
    factors = dict(history=base, identity=identity, names=names, countries=countries, original_colors=assigned, order=order, assignment=assignment, shared=assigned[0] == assigned[1], split_group=f'{identity}_{countries[0]}')
    candidates = list(dict.fromkeys((*colors, 'red', 'yellow', 'blue')))
    add(base, '-', 'story', story, candidates, **factors)
    for target in range(2):
        add(f'{base}_q{target}', base, 'query_story', f'\\nQuestion: What color is the hat belonging to {entities[target]}?\\nAnswer:', candidates, target=target, gold=assigned[target], **factors)
    for updated in range(2):
        uid = f'{base}_u{updated}'
        add(uid, base, 'update', f' {entities[updated]} replaces the hat with a red hat.', candidates, update_target=updated, **factors)
        for target in range(2):
            add(f'{uid}_q{target}', uid, 'query_update', f'\\nQuestion: What color is the hat belonging to {entities[target]}?\\nAnswer:', candidates, target=target, update_target=updated, gold='red' if target == updated else assigned[target], **factors)

(root/'plan.tsv').write_text('\n'.join(rows) + '\n')
(root/'prompts.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in metadata))
(root/'smoke.tsv').write_text('\n'.join(rows[:9]) + '\n')
print(f'{len(cases)} histories, {len(rows)} segments, {sum(m["kind"].startswith("query") for m in metadata)} queries')
