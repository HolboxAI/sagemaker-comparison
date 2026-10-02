#!/usr/bin/env python3
"""Download the account's prior evaluator source + one dataset JSONL as format reference."""
import os, io, json, tarfile, urllib.request
from fireworks import Fireworks

fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="purchase-j087ceepz9i")
os.makedirs("reference/evaluator_orena", exist_ok=True)

# 1. evaluator source tarball
r = fw.evaluators.get_source_code_endpoint("evaluation-orena-focus-reward")
urls = r.filename_to_signed_urls or r.filenameToSignedUrls
for fn, url in urls.items():
    data = urllib.request.urlopen(url).read()
    with open("reference/evaluator_orena.tar.gz", "wb") as f:
        f.write(data)
    print(f"[evaluator] {fn} -> {len(data)} bytes")
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        names = tar.getnames()
        print("  tarball contents:")
        for n in names:
            print("    -", n)
        tar.extractall("reference/evaluator_orena")
    except Exception as e:
        print("  (not gzip tar?)", type(e).__name__, str(e)[:150])
    break

# 2. dataset JSONL (eval-200, smallest)
r2 = fw.datasets.get_download_endpoint("orena-focus-eval-200")
urls2 = r2.filename_to_signed_urls or r2.filenameToSignedUrls
for fn, url in urls2.items():
    data = urllib.request.urlopen(url).read()
    with open("reference/orena_eval_unfiltered.jsonl", "wb") as f:
        f.write(data)
    print(f"\n[dataset] {fn} -> {len(data)} bytes")
    txt = data.decode("utf-8", errors="replace")
    print("  first 3 lines:")
    for i, line in enumerate(txt.splitlines()[:3]):
        print(f"  [{i}]", line[:800])
    break
