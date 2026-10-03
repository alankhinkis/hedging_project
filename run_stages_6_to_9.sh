#!/usr/bin/env bash
# Stages 6-9 only. Stages 1-5 are already complete and their artifacts are on disk
# (universe, prices, option candidates, selected contracts, Pass B paths, vol panel),
# so re-running them would cost ~25 minutes of redundant GARCH for no change.
#
# Stops on ANY non-zero exit: a syntax error, a missing input and a failed checkpoint all
# exit non-zero, and continuing past any of them lets a later stage consume stale inputs.
set -u
log() { echo; echo "############ $* ############"; date '+%Y-%m-%d %H:%M:%S'; }

for stage in \
  "06_run_hedge.py" \
  "07_analysis.py" \
  "08_spx_anchor.py" \
  "09_robustness.py"
do
  log "STAGE $stage"
  python "scripts/$stage"
  rc=$?
  echo "--> $stage exit $rc"
  if [ $rc -ne 0 ]; then echo "FAILURE in $stage (exit $rc); stopping."; exit $rc; fi
done

log "ALL STAGES DONE"
du -sh data/
