# Project Behavioral Rules & Preferences

- **GPU Execution**: Always run model training and evaluation scripts on available GPUs (Apple MPS on Mac / CUDA on NVIDIA).
- **GPU Resource Cap**: Always enforce a strict upper limit of **<80% of total GPU memory and compute usage** (e.g. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.80` or `torch.cuda.set_per_process_memory_fraction(0.80)`).
