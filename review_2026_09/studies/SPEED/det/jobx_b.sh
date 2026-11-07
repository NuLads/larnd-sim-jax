bash /sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/det/inner_DETX1b.sh; r0=$?; bash /sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/det/inner_DETX0b.sh; exit $(( r0 | $? ))
