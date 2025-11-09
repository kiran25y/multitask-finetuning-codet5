"""
# first time (full run)
python build_pandas_datasets.py --repo-url https://github.com/pandas-dev/pandas

# reuse an existing clone
python build_pandas_datasets.py --no-clone

# useful options
#   --max-commits 4000       # more/less repair mining
#   --depth 0                # 0=full history; >0 shallow clone
#   --steps summarization signature search repair   # subset
#   --skip-clean             # only raw JSONLs (no clean_data)






"""