#!/bin/bash
# Speed bench (20 Oct): the production 10 cm fit KN10S1 (inner_sci_full_KN10S1.sh: own argv + env), shortened to
# --max_nbatch 10 --iterations 300 --save_freq 100 (10 batch shapes -> 10 compiles, then 29 epochs steady), new test name.
# usage: make_bench.sh <name> [extra export lines...]   -> writes inner_<name>.sh
n=$1; shift
src=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_KN10S1.sh
{ grep -v '^python3 ' $src; for e in "$@"; do echo "export $e"; done
  grep '^python3 ' $src | sed -e 's/--iterations 10000/--iterations 300/' -e 's/--max_nbatch 100 /--max_nbatch 10 /' \
      -e 's/--save_freq 200/--save_freq 100/' -e "s/--test_name sci_full_KN10S1 /--test_name sci_full_$n /"; } > inner_$n.sh
