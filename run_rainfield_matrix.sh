#!/usr/bin/env bash
# run_rainfield_matrix.sh: Phase 2 sweep, advecting EXCELL rain cell.
#
# Phase 1 (run_millicar_matrix.sh) held rain uniform and static. This sweep varies
# the cell size while holding the platoon-experienced mean rain rate fixed at the
# nominal value, so differences come from rain variability rather than from a
# different amount of rain.
#
# Cell size is swept in units of the Capsoni et al. (1987) eq. (19) mean cell
# radius rho_0_bar(R_M), not in absolute km, because rho_0 is exponentially
# distributed about that mean (eq. 18) and the mean itself depends on peak rate:
#   0.25 -> small cell, strong gradient across the platoon (22% of cells are smaller)
#   0.50 -> below-typical
#   1.00 -> typical cell
#   2.00 -> large cell, near-uniform limit; consistency check against Phase 1
#
# Tag: s<seed>_<regime>_<scen>_<rain>_c<factor>_<dir>   (distinct from Phase 1 tags,
# whose analyzer regex anchors at the rain value, so the two sets never collide).
#
# rain=0 is not swept here: with no rain the field is identical to Phase 1, so the
# uniform rain=0 runs serve as the dry control.
#
# Usage:
#   SEEDS=10 JOBS=12 ./run_rainfield_matrix.sh
#   FACTORS="0.25 2.0" SEEDS=2 ./run_rainfield_matrix.sh    # quick subset
set -uo pipefail
PROJ="$(cd "$(dirname "$0")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns3-mmwave-6.0}"   # path to your ns3-mmwave 6.0 tree
OUTDIR="$PROJ/results_rainfield"
NUMVEH="${NUMVEH:-8}"
SIMTIME="${SIMTIME:-30}"
SEEDS="${SEEDS:-10}"
REGIMES="${REGIMES:-fixwidth}"
RAINS="${RAINS:-20 37.5 50}"
SCENARIOS="${SCENARIOS:-baseline attackA attackB}"
FACTORS="${FACTORS:-0.25 0.5 1.0 2.0}"
CELLDIR="${CELLDIR:-cross}"
CELLSPEED="${CELLSPEED:-10}"
JOBS="${JOBS:-12}"

mkdir -p "$OUTDIR"
[[ -x "$NS3_DIR/ns3" ]] || { echo "ERROR: ns3 not found at $NS3_DIR"; exit 1; }

run_one () {
  local seed=$1 regime=$2 scen=$3 rain=$4 factor=$5
  local tag="s${seed}_${regime}_${scen}_${rain}_c${factor}_${CELLDIR}"
  local pkt="$OUTDIR/pkts-${tag}.csv"
  [[ -s "$pkt" ]] && { printf "  • %-46s skip\n" "$tag"; return; }
  local log; log=$(mktemp)
  ( cd "$NS3_DIR" && ./ns3 run --no-build \
      "weather-v2x-millicar --scenario=$scen --trueRainRate=$rain --numVehicles=$NUMVEH \
       --simTime=$SIMTIME --regime=$regime --rngRun=$((seed+1)) --simTag=$tag \
       --outputDir=$OUTDIR/ --logPkts=1 \
       --rainField=cell --cellSizeFactor=$factor --cellDir=$CELLDIR --cellSpeed=$CELLSPEED" \
  ) >"$log" 2>&1
  if [[ -s "$pkt" ]]; then
    printf "  ✓ %-46s tx=%s rx=%s\n" "$tag" "$(grep -c '^tx,' "$pkt")" "$(grep -c '^rx,' "$pkt")"
  else
    printf "  ✗ %-46s FAILED (see %s)\n" "$tag" "$log"; tail -3 "$log" | sed 's/^/      /'
  fi
  rm -f "$log"
}

total=$(( SEEDS * $(wc -w <<<"$REGIMES") * $(wc -w <<<"$SCENARIOS") * $(wc -w <<<"$RAINS") * $(wc -w <<<"$FACTORS") ))
echo "Rain-field matrix: factors=[$FACTORS] dir=$CELLDIR @ ${CELLSPEED}m/s"
echo "  regimes=[$REGIMES] scenarios=[$SCENARIOS] rains=[$RAINS] seeds=0..$((SEEDS-1)) -> $total runs, JOBS=$JOBS"
for seed in $(seq 0 $((SEEDS-1))); do
  for regime in $REGIMES; do
    for scen in $SCENARIOS; do
      for rain in $RAINS; do
        for factor in $FACTORS; do
          while [[ $(jobs -rp | wc -l | tr -d ' ') -ge $JOBS ]]; do sleep 1; done
          run_one "$seed" "$regime" "$scen" "$rain" "$factor" &
        done
      done
    done
  done
done
wait
echo "Done -> $OUTDIR"
