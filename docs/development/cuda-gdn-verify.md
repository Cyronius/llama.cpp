# CUDA GDN verification kernel

`GGML_CUDA_GDN_L8=1` enables an alternate gated-delta-net kernel for 8-16 token verification batches with saved recurrent states. The default is off. Unset the variable or use `GGML_CUDA_GDN_L8=0` for the existing kernel. Set the variable before starting the process.

The alternate kernel requires NVIDIA Ampere or later device code, a 32-lane warp, state size 128, scalar gates, and more than one snapshot slot. Other shapes, KDA, HIP and MUSA use the existing implementation. It supports activated and raw gates, strided q/k/v inputs, and indexed state gathering.

Eight lanes cooperate on each state column. Each lane holds the partial sums of four lanes from the existing kernel; the reduction preserves its xor 16, 8, 4, 2, 1 addition tree. All k/v/gate inputs for the verification batch are staged in shared memory before recurrence. The kernel writes the ordinary eager snapshots and uses the existing rollback layout.

## Provenance

Adapted from ArthurOstapenko's `gated_delta_net_l8v_cuda` and `gdn_l8_reduce` in the [Yukon CUDA engine](https://github.com/Layr-Labs/cudafast-bonsai2-9b-engine/tree/36c95362b23b83724503fef4e88e61d748f9355f). The source MIT notice is in [licenses/yukon-cudafast](../../licenses/yukon-cudafast).

This port contains the eager-snapshot verification kernel. It does not adopt the source's lazy state encoding, prefill kernel, cache eviction policy, matrix multiplication changes, graph fusions, DFlash selector, or benchmark harness. Source dispatch was specific to DGX Spark; the portable Ampere instructions here remain opt-in while other devices are measured.

## Validation

The backend test suite includes the dispatch boundaries 7/8/15/16/17 tokens, strided tensors, multiple sequences, indexed state reads, raw gates, partial snapshot windows, final-state-only and KDA fallbacks:

```sh
GGML_CUDA_GDN_L8=0 ./build/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET
GGML_CUDA_GDN_L8=1 ./build/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET
```

Keep performance claims specific to the measured shapes and device. The source leaderboard compares a full speculative runtime against a serial control; it does not isolate this kernel or compare against the current fork.

### H200 screen (2026-10-17)

A Release sm_90 build passed all 56 GATED_DELTA_NET backend cases with the option off and on. A direct original/candidate comparison passed 432 bytewise attention and snapshot checks with production `-use_fast_math`, three fresh seeds, token counts 8/9/15/16, 12/32/48 value heads, one/two sequences, snapshot counts 3/17/18, and activated/raw gates. A compile-only sm_75 check passed; no sm_75 execution is claimed.

Nine staged Bonsai 2 variants (2B, 4B and 9B in PTQ1_0, PQ2_0 and Q2_0) passed bytewise full-logit comparisons over 16 corpus slices of 512 accepted tokens each. The screen alternated 8/15/16-token batches and rolled back partial 15-token batches. Nsight Systems confirmed `gated_delta_net_l8v_cuda` execution for every variant. These are staged-artifact checks, not validation of every current published release, MTP integration or a complete speculative serving loop.

A staged 27B Q2_0 variant also passed the same checks over four 512-token slices, with kernel execution confirmed. A 27B PTQ1_0 comparison was inconclusive because a harness adjustment gave the two runs different slice counts; it is excluded. Published MTP variants, 27B PQ2_0/F16 and DGX Spark remain untested.

At 32 value heads, one sequence and 18 snapshot slots, median original/candidate kernel time was 1.219x for activated gates and 1.214x for raw gates. At 48 heads and two sequences it fell to 0.864x and 0.875x. These CUDA-event measurements cover the kernel alone; they are not generation-throughput gains. The multi-sequence regression and unmeasured DGX Spark behavior are reasons to keep the option off by default.

A short 9B PQ2 full-logit verification/rollback screen was noisy and did not establish a model-level speedup: four runs used ABBA ordering, with off times 9.601/12.019 seconds and on times 9.609/13.492 seconds. Only two samples per arm were collected. The option remains experimental.
