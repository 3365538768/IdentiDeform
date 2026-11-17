#!/bin/bash
# v3g job wrapper. <name> = <scene>_v11 / <scene>_v3t0 (official run) or <scene>_v11_hy (hybrid: dynamic stage only,
# physical params initialised from our calibrated prediction (bgsub/offset ensemble), static+velocity ckpts copied from <scene>_v11)
# or <scene>_v11_rs (control: same as _hy but default init, i.e. dynamic-stage restart).
set -u
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test; cd $R
name=$1; gpu=$2
case $name in
  *_hy|*_rs)
    src=${name%_*}; scene=${name%%_*}
    until [ -e runs/v3g/pacnerf/$src/ckpt/model/train_velocity.pt ] && /usr/bin/grep -q "dynamic Train" docs/v3g/logs/pn_$src.log 2>/dev/null; do sleep 60; done
    source scripts/v3g/pn_env.sh
    if [ ! -e runs/v3g/pacnerf/$name/hybrid_init.json ]; then
      if [ "${name##*_}" = hy ]; then
        args=$(python3 -I -c "import json,sys;d=json.load(open('docs/v3g/pn_results/OURS.json'))['methods']['bgsub/offset'][sys.argv[1]]['estimate'];print(d['E'],d['nu'],d.get('tau_y','keep'))" $scene)
      else
        args="keep keep keep"
      fi
      $PNV/bin/python -I scripts/v3g/pn_hybrid_prep.py $src $name $args > docs/v3g/logs/pn_${name}_prep.log 2>&1 || exit 1
    fi
    cfg=scripts/v3g/pn_cfgs/$name.py
    sed -e "s#/${src}/ckpt#/${name}/ckpt#" -e "s#override of the official#(dynamic-only rerun) override of the official#" scripts/v3g/pn_cfgs/$src.py > $cfg
    ;;
esac
# retry on CUDA OOM caused by other tenants of the shared GPU (up to 3 attempts)
for att in 1 2 3; do
  bash scripts/v3g/pn_run.sh $name $gpu
  tail -1 docs/v3g/logs/pn_$name.log | /usr/bin/grep -q "rc=0" && break
  /usr/bin/grep -q "out of memory" docs/v3g/logs/pn_$name.log || break
  # the official dynamic-stage resume skips the interrupted sub-stage, so restart the dynamic stage from scratch
  /usr/bin/grep -q "dynamic Train" docs/v3g/logs/pn_$name.log && rm -f runs/v3g/pacnerf/$name/ckpt/model/train_dynamic.pt
  echo "$(date -u +%FT%TZ) $name OOM attempt $att, retrying" >> docs/v3g/logs/pn_retries.log; sleep 180
done
