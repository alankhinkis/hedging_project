#!/usr/bin/env bash
# Full-sample rebuild, 1996-2023. Stops on a hard failure (exit >= 2 = missing input or no
# WRDS); continues past a checkpoint failure (exit 1), which is informational, but records it.
set -u
log() { echo; echo "############ $* ############"; date '+%Y-%m-%d %H:%M:%S'; }
FAILED=""
for stage in \
  "01_build_universe.py" \
  "02_pull_crsp.py" \
  "03_pull_options.py" \
  "04_select_contracts.py" \
  "05_build_vol.py" \
  "06_run_hedge.py" \
  "07_analysis.py" \
  "08_spx_anchor.py" \
  "09_robustness.py"
do
  log "STAGE $stage"
  python "scripts/$stage"
  rc=$?
  echo "--> $stage exit $rc"
  # Stop on ANY non-zero exit. A syntax error, a missing input and a failed checkpoint all
  # exit non-zero, and continuing past any of them means later stages silently consume stale
  # inputs -- which is exactly what happened on the first attempt: stage 02 died on a syntax
  # error, the runner treated it as an informational checkpoint failure, and stage 03 started
  # building on the previous sample window's calendar.
  if [ $rc -ne 0 ]; then echo "FAILURE in $stage (exit $rc); stopping."; exit $rc; fi
done
log "ALL STAGES DONE"
[ -n "$FAILED" ] && echo "checkpoint failures (informational):$FAILED"
du -sh data/
exit 0
