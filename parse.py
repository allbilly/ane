import struct
import sys
from hwx_parsing import parse_hwx

with open(sys.argv[1], 'rb') as f:
    hwx = f.read()
    
# Parse from offset 0x4000 (tsk_start)
correct_data = hwx[0x4000:]
print('=== Parsing sum.hwx from offset 0x4000 (correct) ===')
parse_hwx(correct_data, subtype=4, dump_json=False)