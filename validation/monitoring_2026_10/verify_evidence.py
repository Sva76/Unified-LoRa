"""Verify published evidence; optional offline replay makes no training calls."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name, 'Unsafe archive path')
    return p


def verify():
    index = json.loads((ROOT / 'evidence_index.json').read_text())
    bundles = {}
    for row in index['bundles']:
        packed = (ROOT / safe_path(row['archive'])).read_bytes()
        require(digest(packed) == row['archive_sha256'], 'Archive checksum mismatch')
        raw = gzip.decompress(packed)
        require(digest(raw) == row['source_sha256'], 'Original JSON checksum mismatch')
        require(len(raw) == row['source_bytes'], 'Source size mismatch')
        bundle = json.loads(raw)
        require(set(bundle['files']) == set(bundle['sha256']), 'Missing file checksums')
        require(len(bundle['files']) == row['embedded_files'], 'File count mismatch')
        for name, content in bundle['files'].items():
            safe_path(name)
            require(isinstance(content, str), 'Non-text embedded file')
            require(digest(content.encode('utf-8')) == bundle['sha256'][name],
                    'Embedded file checksum mismatch: ' + name)
        require(row['id'] not in bundles, 'Duplicate bundle ID')
        bundles[row['id']] = bundle
    for copy in index['exposed_copies']:
        data = (ROOT / safe_path(copy['path'])).read_bytes()
        original = bundles[copy['bundle']]['files'][copy['embedded_path']].encode('utf-8')
        require(data == original and digest(data) == copy['sha256'],
                'Exposed copy differs from archive: ' + copy['path'])
    print('Verified %d archives and %d embedded files.' %
          (len(bundles), sum(len(b['files']) for b in bundles.values())))
    return bundles


def extract_one(bundle, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in bundle['files'].items():
        path = destination / safe_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(content.encode('utf-8'))


def replay(bundles):
    programs = {
        'stability_104': '''import json
import analyze104
result = analyze104.analyze('runs/104')
expected = json.load(open('esito104.json'))
if result != expected: raise ValueError('Seed 104 replay mismatch')
print('Seed 104: exact result replay; Spearman', result['primary_spearman'])
''',
        'mechanics_107': '''import json
import collector107
manifest = json.load(open('runs/107/manifest.json'))
result = collector107.analyze('runs/107', manifest)
expected = json.load(open('esito107.json'))
if result != expected: raise ValueError('Seed 107 replay mismatch')
print('Seed 107: exact prospective decision replay;', result['episode_count'], 'episodes')
''',
    }
    with tempfile.TemporaryDirectory(prefix='unified-lora-replay-') as temp:
        for key, program in programs.items():
            out = Path(temp) / key
            extract_one(bundles[key], out)
            subprocess.run([sys.executable, '-B', '-c', program], cwd=out, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--extract', type=Path,
                        help='New destination for all embedded files; never overwritten')
    args = parser.parse_args()
    bundles = verify()
    if args.extract:
        args.extract.mkdir(parents=True, exist_ok=False)
        for key, bundle in bundles.items():
            extract_one(bundle, args.extract / key)
        print('Extracted all bundles to', args.extract)
    if args.replay:
        replay(bundles)


if __name__ == '__main__':
    main()
