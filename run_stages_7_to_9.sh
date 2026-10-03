#!/usr/bin/env bash
# Stages 7-9. Stage 06's results are already on disk and its checkpoint passes, so re-running
# it would cost ~30 minutes for an identical hedge_results.parquet.
#
# Each stage is a separate python invocation, so memory is released between them -- which
# matters here: the 28-year panels are large and the machine ran low on RAM during the
# previous attempt.
set -u
log() { echo; echo "############ $* ############"; date '+%Y-%m-%d %H:%M:%S'; }

for stage in "07_analysis.py" "08_spx_anchor.py" "09_robustness.py"; do
  log "STAGE $stage"
  python "scripts/$stage"
  rc=$?
  echo "--> $stage exit $rc"
  if [ $rc -ne 0 ]; then echo "FAILURE in $stage (exit $rc); stopping."; exit $rc; fi
done

log "ALL STAGES DONE"
du -sh data/
