#!/bin/bash
# ENDFIT (b) live check on the 5 chain-head fits: per batch, constraint tracks + FROZEN tracks = batch size.
G=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs
F="$G/EZT1-41873499.out $G/EZT2-41873503.out $G/EZT4-41873508.out $G/EZS1-41873512.out $G/EZS2-41873517.out"
for i in $(seq 1 30); do
  ok=1; for f in $F; do c=$(grep -c "endpoint drift zero-mean constraint" $f 2>/dev/null); [ "${c:-0}" -ge 4 ] || ok=0; done
  [ $ok = 1 ] && break; sleep 120
done
for f in $F; do
  echo "== $(basename $f): $(grep -c 'endpoint drift zero-mean constraint' $f) constraint lines"
  grep -o "batch [0-9]*: drift-direction geometry FROZEN on [0-9]*/[0-9]*\|batch [0-9]*: endpoint drift zero-mean constraint on [0-9]*/[0-9]* tracks (sum w^2 [0-9.]*)" $f \
  | awk '/FROZEN/{split($NF,a,"/"); b=$2; fz[b]=a[1]; T[b]=a[2]} /zero-mean/{split($8,a,"/"); b=$2; u[b]=a[1]; T2[b]=a[2]; w[b]=$NF}
         END{n=0; bad=0; for (b in u){n++; if (u[b]+fz[b]!=T2[b] || T[b]!=T2[b]) {bad++; print "  MISMATCH batch " b " used " u[b] " frozen " fz[b] " of " T2[b]}; su+=u[b]; st+=T2[b]}
             printf "  %d batches checked, %d mismatches; constraint on %d of %d tracks (%.0f%%)\n", n, bad, su, st, 100*su/st}'
  grep -m3 "Traceback\|Error\|nan" $f | grep -v "cuda\|CUDA\|plugin" | head -3
done
