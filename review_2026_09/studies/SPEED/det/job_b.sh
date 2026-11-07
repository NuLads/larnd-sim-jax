bash /sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/det/inner_DET1b.sh; r0=$?; bash /sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/det/inner_DET0b.sh; exit $(( r0 | $? ))
