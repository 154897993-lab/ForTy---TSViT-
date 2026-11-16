# GPU performance benchmark

- device: cuda, total memory: 47.4 GB
- warmup: 2, measured steps: 10

| batch | mode | samples/s | step ms | augment ms | peak MB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | clean | 1108.2 | 7.2 | 0.0 | 273.8 |
| 8 | independent | 13.2 | 606.0 | 591.0 | 273.8 |
| 16 | clean | 1075.2 | 14.9 | 0.0 | 522.3 |
| 16 | independent | 13.3 | 1207.3 | 1187.6 | 522.3 |
| 32 | clean | 1094.2 | 29.2 | 0.0 | 1017.3 |
| 32 | independent | 15.6 | 2057.1 | 2025.9 | 1017.3 |

## Observed epoch times

| run | epochs | mean s | median s | 100-epoch h |
| --- | ---: | ---: | ---: | ---: |
| clean_D_b8 | 100 | 4.4 | 4.2 | 0.12 |
| cloud_B_ind_b8 | 100 | 38.0 | 37.3 | 1.05 |
| cloud_B_ind_b16 | 100 | 37.6 | 37.2 | 1.05 |
| cloud_B_ind_b32 | 100 | 37.6 | 37.0 | 1.05 |
