#!/usr/bin/env bash
# usage: scripts/v3v/chain3.sh <obj>   LOO-damping confirmation: run3 then eval3 (a800-2 GPU 7)
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
bash $R/scripts/v3v/run_bg.sh confirm_$1 run3 --objs $1 && bash $R/scripts/v3v/run_bg.sh confirm_$1 eval3 --objs $1
