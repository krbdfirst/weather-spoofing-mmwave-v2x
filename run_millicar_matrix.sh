#!/usr/bin/env bash
# run_millicar_matrix.sh: Phase 1 sweep, weather-spoofing study on MilliCar.
#
# Sweeps regime x scenario x rain x seed (RNG run) over the weather-v2x-millicar
# scenario in the ns3-mmwave-6.0 tree. Each run writes, into results/:
#   pkts-s<seed>_<regime>_<scen>_<rain>.csv  per-packet tx/rx (seq,bytes,latency,dist)
#   phy-s<seed>_<regime>_<scen>_<rain>.csv   per-rx-TB SINR/MCS/TB-size
#   meta-s<seed>_<regime>_<scen>_<rain>.csv  run metadata (band, bandwidth)
# Two regimes: fixwidth (100 MHz all bands) and realwidth (per-band deployable BW).
# Resumable: an existing pkts file is skipped. Parallel via JOBS.
#
# Usage: SEEDS=1 regimes="fixwidth realwidth" RAINS="0 10 20 30 37.5 50" JOBS=6 ./run_millicar_matrix.sh
set -uo pipefail
PROJ="$(cd "$(dirname "$0")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns3-mmwave-6.0}"   # path to your ns3-mmwave 6.0 tree
OUTDIR="$PROJ/results"
NUMVEH="${NUMVEH:-8}"
SIMTIME="${SIMTIME:-30}"
SEEDS="${SEEDS:-1}"
REGIMES="${REGIMES:-fixwidth realwidth}"
RAINS="${RAINS:-0 10 20 30 37.5 50}"
SCENARIOS="${SCENARIOS:-baseline attackA attackB}"
JOBS="${JOBS:-6}"

mkdir -p "$OUTDIR"
[[ -x "$NS3_DIR/ns3" ]] || { echo "ERROR: ns3 not found at $NS3_DIR"; exit 1; }

run_one () {
  local seed=$1 regime=$2 scen=$3 rain=$4
  local tag="s${seed}_${regime}_${scen}_${rain}"
  local pkt="$OUTDIR/pkts-${tag}.csv"
  [[ -s "$pkt" ]] && { printf "  • %-30s skip\n" "$tag"; return; }
  local log; log=$(mktemp)
  ( cd "$NS3_DIR" && ./ns3 run --no-build \
      "weather-v2x-millicar --scenario=$scen --trueRainRate=$rain --numVehicles=$NUMVEH \
       --simTime=$SIMTIME --regime=$regime --rngRun=$((seed+1)) --simTag=$tag --outputDir=$OUTDIR/ --logPkts=1" \
  ) >"$log" 2>&1
  if [[ -s "$pkt" ]]; then
    printf "  ✓ %-30s tx=%s rx=%s\n" "$tag" "$(grep -c '^tx,' "$pkt")" "$(grep -c '^rx,' "$pkt")"
  else
    printf "  ✗ %-30s FAILED (see %s)\n" "$tag" "$log"; tail -3 "$log" | sed 's/^/      /'
  fi
  rm -f "$log"
}

echo "MilliCar matrix: regimes=[$REGIMES] scenarios=[$SCENARIOS] rains=[$RAINS] seeds=0..$((SEEDS-1)) numVeh=$NUMVEH simTime=${SIMTIME}s JOBS=$JOBS"
for seed in $(seq 0 $((SEEDS-1))); do
  for regime in $REGIMES; do
    for scen in $SCENARIOS; do
      for rain in $RAINS; do
        while [[ $(jobs -rp | wc -l | tr -d ' ') -ge $JOBS ]]; do sleep 1; done
        run_one "$seed" "$regime" "$scen" "$rain" &
      done
    done
  done
done
wait
echo "Done -> $OUTDIR"
