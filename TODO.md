- The bug is pre-existing in vLLM main - not introduced by PR-31740. Both main and PR-31740 are missing o_data_type.
  The FlashInfer plan() function added o_data_type parameter, but the vLLM fast_plan_decode wasn't updated to pass it, causing positional argument shift.
  This bug would affect anyone using this code path with the current FlashInfer version. The mxfp4_wip branch correctly added None, # o_data_type to fix it.
- Ensure non_blocking is a boolean (older PyTorch accepts None, newer versions don't) <- flashinfer
- TRTLLM exclusion on sm121 for attention when using flashinfer backend
