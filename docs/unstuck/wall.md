# diversity_bench: ckpt_8002732032.pt @ 8,002,732,032  N=64  spawn seed 0

knob sigma = view sigma x (1+T), logits / (1+T); knob eps = each head uniform with p = min(0.5, 0.05 T). Progress = order-only corridor (window 16, corridor 1500) in route units; wall = 205,440 u; spread = RMS from the medoid of the alive rollouts (u); branches = single-linkage clusters at 512 u; cells = distinct 256 u position cells, novel = count 0 in the checkpoint's int_counts, rare = under 100; same-line = within 100 u of the greedy trajectory for the whole life.

| knob | T | prog mean | prog max | past wall | fin | len med s | spread 25/50/75 % | spread 60 s | branches end/30s/60s | cells | novel | rare | same-line life/60s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 0 | 205,252 | 205,252 | 0% | 0 | 68.5 | 0/0/0 | 0 (64 alive) | 1/1/1 | 1112 | 0 | 0 | 100% / 100% |
| sigma | 0 | 177,320 | 205,440 | 0% | 0 | 71.4 | 388.7/803.3/1,254 | 1,309 (52 alive) | 10/1/1 | 3247 | 0 | 0 | 2% / 2% |
| sigma | 0.25 | 147,054 | 205,454 | 2% | 0 | 72.7 | 481.8/1,022/2,238 | 2,618 (41 alive) | 18/2/7 | 3483 | 0 | 0 | 5% / 5% |
| sigma | 0.5 | 61,638 | 203,666 | 0% | 0 | 13.0 | 241.0/847.3/323.4 | 2,355 (9 alive) | 32/4/7 | 3049 | 0 | 11 | 3% / 3% |
| sigma | 1 | 11,646 | 49,152 | 0% | 0 | 6.34 | 158.0/361.0/595.7 | n/a (0 alive) | 13/0/0 | 620 | 0 | 0 | 2% / 2% |
| sigma | 2 | 4,216 | 15,001 | 0% | 0 | 4.74 | 38.1/204.3/1,088 | n/a (0 alive) | 7/0/0 | 169 | 0 | 0 | 0% / 0% |
| sigma | 4 | 2,436 | 3,995 | 0% | 0 | 6.14 | 64.8/327.8/704.5 | n/a (0 alive) | 2/0/0 | 90 | 0 | 0 | 0% / 0% |
| eps | 0 | 177,320 | 205,440 | 0% | 0 | 71.4 | 388.7/803.3/1,254 | 1,309 (52 alive) | 10/1/1 | 3247 | 0 | 0 | 2% / 2% |
| eps | 0.25 | 19,704 | 102,044 | 0% | 0 | 9.21 | 172.0/1,169/1,986 | n/a (0 alive) | 20/3/0 | 1332 | 0 | 3 | 2% / 2% |
| eps | 0.5 | 9,132 | 50,165 | 0% | 0 | 6.37 | 195.2/481.6/1,214 | n/a (0 alive) | 15/0/0 | 583 | 0 | 31 | 3% / 3% |
| eps | 1 | 5,530 | 15,446 | 0% | 0 | 6.48 | 197.4/563.1/1,592 | n/a (0 alive) | 7/0/0 | 188 | 0 | 3 | 0% / 0% |
| eps | 2 | 3,470 | 7,401 | 0% | 0 | 6.39 | 196.2/579.4/1,338 | n/a (0 alive) | 4/0/0 | 128 | 0 | 1 | 2% / 2% |
| eps | 4 | 2,530 | 3,758 | 0% | 0 | 7.68 | 109.6/832.0/650.6 | n/a (0 alive) | 1/0/0 | 84 | 0 | 0 | 0% / 0% |
