import argparse
import os
import shutil
import sys
import tarfile
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.cub200 import cub_root, validate_cub_root

URLS = [
    'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1',
    'http://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1',
]
MIN_EXPECTED_BYTES = 500 * 1024 * 1024  # CUB tgz is far larger than 88MB; catch truncated downloads early.


def is_valid_archive(path):
    if not os.path.exists(path):
        return False, 'archive does not exist'
    size = os.path.getsize(path)
    if size < MIN_EXPECTED_BYTES:
        return False, 'archive too small: {} bytes'.format(size)
    try:
        with tarfile.open(path, 'r:gz') as tar:
            # Read member table; this catches most truncated gzip/tar streams.
            members = tar.getmembers()
        if not members:
            return False, 'archive has no members'
    except Exception as e:
        return False, str(e)
    return True, 'ok'


def download(url, dst):
    tmp = dst + '.part'
    if os.path.exists(tmp):
        os.remove(tmp)
    print('Downloading:', url)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length', 0) or 0)
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print('\r{:.1f}% ({:.1f}/{:.1f} MB)'.format(done * 100.0 / total, done / 1048576, total / 1048576), end='')
            else:
                print('\r{:.1f} MB'.format(done / 1048576), end='')
    print('\nDownloaded bytes:', os.path.getsize(tmp))
    os.replace(tmp, dst)
    print('Saved:', dst)


def main():
    p = argparse.ArgumentParser(description='Prepare CUB-200-2011 under data/CUB_200_2011')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--check_only', action='store_true')
    p.add_argument('--force_download', action='store_true', help='Delete existing archive/partial extracted directory and download again.')
    args = p.parse_args()
    root = cub_root(args.data_root)
    archive = os.path.join(args.data_root, 'CUB_200_2011.tgz')
    print('CUB root:', root)

    if os.path.exists(root) and not args.force_download:
        try:
            validate_cub_root(root, require_attributes=True)
            print('CUB-200-2011 check passed: 200 classes, 11788 images.')
            return
        except Exception as e:
            if args.check_only:
                raise
            print('Existing CUB directory is incomplete or invalid:', e)
            print('Use --force_download to remove it and retry, or manually fix:', root)
            raise SystemExit(1)

    if args.check_only:
        raise SystemExit('CUB data not found/valid. Please place the official CUB_200_2011 directory at {}'.format(root))

    os.makedirs(args.data_root, exist_ok=True)
    if args.force_download:
        if os.path.exists(archive):
            print('Removing old archive:', archive)
            os.remove(archive)
        part = archive + '.part'
        if os.path.exists(part):
            print('Removing partial archive:', part)
            os.remove(part)
        if os.path.exists(root):
            print('Removing incomplete extracted directory:', root)
            shutil.rmtree(root)

    ok, reason = is_valid_archive(archive)
    if not ok:
        if os.path.exists(archive):
            print('Existing archive is invalid:', reason)
            print('Removing invalid archive:', archive)
            os.remove(archive)
        last_err = None
        for url in URLS:
            try:
                download(url, archive)
                ok, reason = is_valid_archive(archive)
                if not ok:
                    raise RuntimeError('Downloaded archive failed validation: {}'.format(reason))
                break
            except Exception as e:
                last_err = e
                print('Download failed:', e)
                if os.path.exists(archive):
                    os.remove(archive)
        else:
            raise SystemExit('Automatic CUB download failed. Manually download CUB_200_2011.tgz and extract it to {}. Last error: {}'.format(root, last_err))
    else:
        print('Using existing valid archive:', archive)

    print('Extracting:', archive)
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(args.data_root)
    except Exception as e:
        raise SystemExit('Extraction failed. The archive is likely incomplete. Re-run with --force_download. Error: {}'.format(e))
    validate_cub_root(root, require_attributes=True)
    print('CUB-200-2011 ready at:', root)


if __name__ == '__main__':
    main()
