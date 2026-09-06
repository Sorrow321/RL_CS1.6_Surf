# diversity_bench: ckpt_8002732032.pt @ 8,002,732,032  N=64  spawn seed 0

knob sigma = view sigma x (1+T), logits / (1+T); knob eps = each head uniform with p = min(0.5, 0.05 T). Progress = order-only corridor (window 16, corridor 1500) in route units; wall = 205,440 u; spread = RMS from the medoid of the alive rollouts (u); branches = single-linkage clusters at 512 u; cells = distinct 256 u position cells, novel = count 0 in the checkpoint's int_counts, rare = under 100; same-line = within 100 u of the greedy trajectory for the whole life.

| knob | T | prog mean | prog max | past wall | fin | len med s | spread 25/50/75 % | spread 60 s | branches end/30s/60s | cells | novel | rare | same-line life/60s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 0 | 205,252 | 205,252 | 0% | 0 | 68.5 | 0/0/0 | 0 (64 alive) | 1/1/1 | 1112 | 0 | 0 | 100% / 100% |
| sigma | 0 | 177,320 | 205,440 | 0% | 0 | 71.4 | 388.7/803.3/1,254 | 1,309 (52 alive) | 10/1/1 | 3247 | 0 | 0 | 2% / 2% |
| sigma | 0.1 | 165,997 | 205,684 | 2% | 0 | 72.1 | 385.5/910.7/1,569 | 1,703 (49 alive) | 11/2/6 | 3480 | 0 | 2 | 8% / 8% |
| sigma | 0.2 | 163,645 | 205,382 | 0% | 0 | 72.8 | 528.7/1,235/2,255 | 2,481 (46 alive) | 20/4/6 | 3434 | 0 | 1 | 0% / 0% |
| sigma | 0.3 | 121,266 | 205,616 | 6% | 0 | 64.6 | 609.5/1,384/2,174 | 3,383 (35 alive) | 26/5/13 | 3657 | 0 | 23 | 3% / 3% |
| sigma | 0.4 | 100,644 | 205,696 | 2% | 0 | 35.6 | 410.9/1,334/999.2 | 3,568 (23 alive) | 38/4/12 | 3751 | 0 | 4 | 2% / 2% |
| sigma | 0.5 | 61,638 | 203,666 | 0% | 0 | 13.0 | 241.0/847.3/323.4 | 2,355 (9 alive) | 32/4/7 | 3049 | 0 | 11 | 3% / 3% |
| sigma | 0.75 | 20,446 | 114,347 | 0% | 0 | 10.5 | 273.9/641.7/682.3 | n/a (0 alive) | 19/3/0 | 1245 | 0 | 7 | 2% / 2% |
