#!/usr/bin/env bash
# usage: scripts/v3v/eval_chain.sh <obj>   eval then eval2 (exploratory damped arm + damped oracle)
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
bash $R/scripts/v3v/run_bg.sh evalc_$1 eval --objs $1 && bash $R/scripts/v3v/run_bg.sh evalc_$1 eval2 --objs $1
