import sys
from hwx_parsing import parse_hwx, load_hwx_data

if len(sys.argv) < 2:
    raise SystemExit(f"Usage: {sys.argv[0]} PATH [SUBTYPE]")

subtype = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ane_data, subtype = load_hwx_data(sys.argv[1], subtype)
if not ane_data:
    raise SystemExit(f"Error: could not identify HWX command stream in {sys.argv[1]}")

print(f"=== Parsing {sys.argv[1]} ===")
parse_hwx(ane_data, subtype=subtype, dump_json=False)
