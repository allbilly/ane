"""Apply expt3 results: comment out UNNEEDED registers in examples_expt/*.py."""
import sys

UNNEEDED_RELU = ['(reg.W1,', '(reg.W2,', '(reg.W3,', '(reg.W5,', '(reg.W7,', '(reg.W9,']

def apply_comments(filename, skip_patterns):
    lines = open(filename).read().split('\n')
    out = []
    for line in lines:
        if any(line.strip().startswith(x) for x in skip_patterns):
            out.append('# UNNEEDED (expt3): ' + line)
        else:
            out.append(line)
    open(filename, 'w').write('\n'.join(out))
    print(f'Applied to {filename}: {len([l for l in out if l.startswith("# UNNEEDED")])} lines commented')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if 'relu' in filename or 'sigmoid' in filename:
            apply_comments(filename, UNNEEDED_RELU)
        print('Unknown file, no changes')
    else:
        apply_comments('examples_expt/relu.py', UNNEEDED_RELU)
