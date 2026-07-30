# Global DINOv3 modes

Run `01_sample_h3_images.py` per city, fit candidate codebooks, review `03_build_mode_gallery.py` output, then select a K with `04_select_mode_model.py`. Assignments and sparse histograms are compared using exact base-2 Jensen--Shannon similarity. Set an explicit threshold for large city cross-products; `--threshold -1` retains every result. The Slurm coordinator must pause after gallery generation unless `SELECTED_K` is explicitly set.
