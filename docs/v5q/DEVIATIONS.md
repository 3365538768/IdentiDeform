# V5q execution log (timing only; no protocol change)

- 2026-11-16 01:23 UTC: protocol written (docs/v5q/PROTOCOL.json); P1 + GT noise-floor launched: a800-2 r_queue (docs/v5q/queue.json, max 5/GPU, GPUs 0-7) and a800-4 pool (GPUs 0-6, 5/GPU; GPU 7 left to the prior agent). P2 held until docs/v5p/HANDOFF.json + gate pass.
- 2026-11-16 01:33 UTC: lead gate decision: main line stays on V2, V5b refinement run as a supplementary experiment (protocol unchanged). docs/v5p/HANDOFF.json final (01:22 UTC, 9-run ensemble val+test); docs/v5q/PRIOR_ENSEMBLE_{val,test}.json built and checked against docs/v5p/ENSEMBLE_BCD9_{VAL,TEST}.json; P2 released and placed ahead of pending P1/P3 (running tasks unaffected); P4 last.
- 2026-11-16 03:45 UTC: stopped on lead instruction (servers available only until ~06:40 UTC). STOP file, queue gpus [], a800-4 pool + finisher killed, all v5q sims killed.
