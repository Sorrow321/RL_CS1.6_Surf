# diversity_bench: ckpt_8002732032.pt @ 8,002,732,032  N=64  spawn seed 0

knob sigma = view sigma x (1+T), logits / (1+T); knob eps = each head uniform with p = min(0.5, 0.05 T). Progress = order-only corridor (window 16, corridor 1500) in route units; wall = 205,440 u; spread = RMS from the medoid of the alive rollouts (u); branches = single-linkage clusters at 512 u; cells = distinct 256 u position cells, novel = count 0 in the checkpoint's int_counts, rare = under 100; same-line = within 100 u of the greedy trajectory for the whole life.

| knob | T | prog mean | prog max | past wall | fin | len med s | spread 25/50/75 % | spread 60 s | branches end/30s/60s | cells | novel | rare | same-line life/60s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 0 | 205,252 | 205,252 | 0% | 0 | 68.5 | 0/0/0 | 0 (64 alive) | 1/1/1 | 1112 | 0 | 0 | 100% / 100% |
| eps | 0.02 | 126,050 | 205,460 | 2% | 0 | 67.0 | 504.5/1,079/2,267 | 3,420 (34 alive) | 27/3/8 | 3614 | 0 | 1 | 8% / 8% |
| eps | 0.05 | 86,904 | 205,411 | 0% | 0 | 21.2 | 705.4/406.7/690.5 | 4,955 (19 alive) | 34/3/11 | 3406 | 0 | 6 | 3% / 3% |
| eps | 0.1 | 46,948 | 198,730 | 0% | 0 | 11.9 | 307.2/1,411/1,603 | 257.0 (2 alive) | 37/5/1 | 2662 | 0 | 7 | 5% / 5% |
| eps | 0.2 | 27,703 | 116,948 | 0% | 0 | 10.1 | 372.9/1,121/2,143 | n/a (0 alive) | 24/7/0 | 1631 | 0 | 12 | 2% / 2% |
