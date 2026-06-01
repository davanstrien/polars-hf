# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars-hf @ git+https://github.com/davanstrien/polars-hf@main",
# ]
# ///
"""Smoke-test polars-hf reads on Hugging Face Jobs.

Installs polars-hf straight from the public GitHub repo (no PyPI release needed)
and reads parquet from a Hugging Face bucket on HF infrastructure. Pass the
HF_TOKEN secret so private buckets can be read:

    hf jobs uv run examples/run_on_hf_jobs.py --secrets HF_TOKEN --flavor cpu-basic
"""

import polars as pl

import polars_hf as plhf

BASE = "hf://buckets/davanstrien/polars-hf-wheels/smoke-test-full"

print(f"polars {pl.__version__} | polars_hf {plhf.__version__}")

# 1. Single-file read.
df = plhf.scan_bucket(f"{BASE}/filtered.parquet").collect()
print("SINGLE", df.shape, df.columns)
assert df.shape == (500, 4), f"expected (500, 4), got {df.shape}"

# 2. Multi-file glob + projection/predicate pushdown on the streaming engine.
out = (
    plhf.scan_bucket(f"{BASE}/bench/run_*.parquet")
    .filter(pl.col("value") > 0.5)
    .group_by("category")
    .agg(pl.len().alias("n"), pl.col("value").mean().round(4).alias("mean_value"))
    .sort("category")
    .collect(engine="streaming")
)
print(out)
assert out.height == 5
assert out["n"].sum() > 0

# 3. @revision must be rejected.
try:
    plhf.scan_bucket(
        "hf://buckets/davanstrien/polars-hf-wheels@main/x.parquet"
    ).collect()
    raise SystemExit("FAIL: @revision should have errored")
except ValueError as e:
    assert "@revision" in str(e), f"unexpected error: {e}"
    print("REJECT ok:", str(e).splitlines()[0])

print("\nALL JOB TESTS PASSED")
