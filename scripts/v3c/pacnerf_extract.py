"""V3c: extract the 4 named PAC-NeRF scenes from data.zip (untrusted) into a NEW empty directory.

Only regular files data/<scene>/all_data.json and data/<scene>/data/r_<v>_<f>.png are written;
names are validated (no absolute paths, no '..', fixed pattern). Nothing from the archive is executed.
Run with python -I.  python3 -I scripts/v3c/pacnerf_extract.py <data.zip> <new_empty_dir>
"""
import re, sys, zipfile
from pathlib import Path

SCENES = ('torus', 'bird', 'playdoh', 'cat')
PATTERN = re.compile(r'^data/(%s)/(all_data\.json|data/r_\d+_-?\d+\.png)$' % '|'.join(SCENES))


def main(archive, target):
    target = Path(target)
    if target.exists() and any(target.iterdir()):
        raise SystemExit('target must be new/empty')
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir() or not PATTERN.match(info.filename):
                continue
            out = (target / info.filename).resolve()
            if not str(out).startswith(str(target.resolve()) + '/'):
                raise SystemExit('path escape: ' + info.filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, 'wb') as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
            n += 1
    print('extracted', n)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
