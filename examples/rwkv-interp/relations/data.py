"""Memory-mapped RWKV relation trajectories. No model load is needed."""
import csv
import json
from pathlib import Path
import numpy as np

FIELDS = ['r', 'w', 'k', 'v', 'a', 'b', 'wkv_read', 'group_norm', 'attn_norm', 'ffn_norm', 'layer_out']

class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        config = dict(line.rstrip('\n').split('\t', 1) for line in (self.path / 'config.tsv').read_text().splitlines())
        self.layers = int(config['layers'])
        self.embd = int(config['embd'])
        self.head_size = int(config['head_size'])
        self.heads = self.embd // self.head_size
        self.selected = [int(x) for x in config['selected_layers'].split(',') if x]
        with (self.path / 'segments.tsv').open() as f:
            self.segments = {r['id']: r for r in csv.DictReader(f, delimiter='\t')}
        self.prompts = {}
        if (self.path / 'prompts.jsonl').exists():
            self.prompts = {r['id']: r for r in map(json.loads, (self.path / 'prompts.jsonl').read_text().splitlines())}

    def _map(self, segment, name, shape):
        path = self.path / segment / f'{name}.f32'
        expected = int(np.prod(shape)) * 4
        if path.stat().st_size != expected:
            raise ValueError(f'{path}: expected {expected} bytes, found {path.stat().st_size}')
        return np.memmap(path, dtype='<f4', mode='r', shape=shape)

    def trajectory(self, segment):
        if not (self.path / segment / 'DONE').exists():
            raise ValueError(f'{segment} is incomplete')
        n = int(self.segments[segment]['n_tokens'])
        d, h, l, s = self.head_size, self.heads, self.layers, len(self.selected)
        vectors = self._map(segment, 'vectors', (n, l, len(FIELDS), h, d))
        vocab = (self.path / segment / 'logits.f32').stat().st_size // (n * 4)
        return dict(
            wkv=self._map(segment, 'wkv', (n + 1, s, h, d, d)),
            shifts=self._map(segment, 'shifts', (n + 1, s, 2, h, d)),
            vectors={name: vectors[:, :, i] for i, name in enumerate(FIELDS)},
            logits=self._map(segment, 'logits', (n, vocab)),
        )

    def boundary(self, segment):
        d, h = self.head_size, self.heads
        packed = self._map(segment, 'boundary', (self.layers, self.embd * (d + 2)))
        return dict(wkv=packed[:, :self.embd*d].reshape(self.layers, h, d, d),
                    shifts=packed[:, self.embd*d:].reshape(self.layers, 2, h, d))

    def tokens(self, segment):
        """Return exact input IDs, positions, and token-piece bytes for one segment."""
        if not hasattr(self, '_tokens'):
            self._tokens = {}
            with (self.path / 'tokens.tsv').open() as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    item = dict(index=int(row['index']), position=int(row['position']),
                                token=int(row['token']), piece=bytes.fromhex(row['piece_hex']))
                    self._tokens.setdefault(row['segment'], []).append(item)
        return self._tokens[segment]
