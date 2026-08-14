import os
import time
import csv
import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def main():
    # ---- Config ----
    model_folder = "/content/Datasets_nnUNet/nnUNet_results/Dataset801_MSDPancreas/lhunetV2MSDPancreasTrainer__nnUNetPlans__3d_fullres"
    input_folder = "/content/Datasets_nnUNet/nnUNet_raw_data/Dataset801_MSDPancreas/imagesTr"
    use_folds = (0,)
    checkpoint_name = "checkpoint_best.pth"
    n_warmup = 1
    n_profiled_cases = 2          # how many cases get the full torch.profiler treatment
    out_csv = "/content/inference_timing_network_only.csv"
    trace_dir = "/content/profiler_traces"
    os.makedirs(trace_dir, exist_ok=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=use_folds, checkpoint_name=checkpoint_name,
    )

    # ---------------------------------------------------------------
    # Forward hooks: accumulate pure network compute time per case.
    # This sums every call to network.forward() (i.e. every sliding
    # window patch), independent of preprocessing / resampling / I/O.
    # ---------------------------------------------------------------
    network_time_accum = {"total": 0.0, "calls": 0}

    def pre_hook(module, inp):
        torch.cuda.synchronize()
        network_time_accum["_t0"] = time.time()

    def post_hook(module, inp, out):
        torch.cuda.synchronize()
        network_time_accum["total"] += time.time() - network_time_accum["_t0"]
        network_time_accum["calls"] += 1

    predictor.network.register_forward_pre_hook(pre_hook)
    predictor.network.register_forward_hook(post_hook)

    # ---- Discover cases ----
    files = sorted(f for f in os.listdir(input_folder) if f.endswith('.nii.gz'))
    case_ids = sorted(set(f.rsplit('_', 1)[0] for f in files))
    channel_key = 'channel_names' if 'channel_names' in predictor.dataset_json else 'modality'
    num_input_channels = len(predictor.dataset_json[channel_key])
    print(f"Found {len(case_ids)} cases, {num_input_channels} channel(s) each\n")

    reader = SimpleITKIO()
    rows, failed = [], []

    for i, case_id in enumerate(case_ids):
        channel_files = [os.path.join(input_folder, f"{case_id}_{c:04d}.nii.gz")
                          for c in range(num_input_channels)]
        if any(not os.path.exists(f) for f in channel_files):
            print(f"Skipping {case_id}: missing file(s)")
            continue

        try:
            data, properties = reader.read_images(channel_files)

            network_time_accum["total"] = 0.0
            network_time_accum["calls"] = 0

            torch.cuda.synchronize()
            t_total_start = time.time()

            # ---- Optionally wrap this case in a full profiler trace ----
            run_profiler = (len(rows) < n_profiled_cases) and (i >= n_warmup)

            if run_profiler:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    with record_function("predict_single_npy_array"):
                        _ = predictor.predict_single_npy_array(
                            data, properties, None, None, False
                        )

                trace_path = os.path.join(trace_dir, f"{case_id}_trace.json")
                prof.export_chrome_trace(trace_path)
                print(f"\n=== Profiler summary: {case_id} ===")
                print(prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=15
                ))
                print(f"Chrome trace saved to {trace_path}\n")
            else:
                _ = predictor.predict_single_npy_array(
                    data, properties, None, None, False
                )

            torch.cuda.synchronize()
            total_s = time.time() - t_total_start
            network_s = network_time_accum["total"]
            n_patches = network_time_accum["calls"]
            other_s = total_s - network_s  # preprocessing + resampling + postprocessing + overhead

        except Exception as e:
            print(f"[{case_id}] FAILED: {type(e).__name__}: {e}")
            failed.append((case_id, str(e)))
            torch.cuda.empty_cache()
            continue

        if i < n_warmup:
            print(f"[{case_id}] warm-up (discarded): total={total_s:.3f}s "
                  f"network={network_s:.3f}s patches={n_patches}")
            continue

        rows.append((case_id, total_s, network_s, other_s, n_patches))
        print(f"[{i+1}/{len(case_ids)}] {case_id}: total={total_s:.3f}s  "
              f"network={network_s:.3f}s ({100*network_s/total_s:.1f}%)  "
              f"other={other_s:.3f}s  patches={n_patches}")

    # ---- Summary ----
    if rows:
        arr = np.array([[r[1], r[2], r[3]] for r in rows])
        labels = ['total', 'network_only', 'other (pre/post/io)']
        print("\n=== Timing summary (seconds) ===")
        print(f"{'stage':<22}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'median':>8}")
        for j, label in enumerate(labels):
            col = arr[:, j]
            print(f"{label:<22}{col.mean():>8.3f}{col.std():>8.3f}"
                  f"{col.min():>8.3f}{col.max():>8.3f}{np.median(col):>8.3f}")

        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case_id', 'total_s', 'network_only_s', 'other_s', 'n_patches'])
            w.writerows(rows)
        print(f"\nSaved to {out_csv}")

    if failed:
        print(f"\n{len(failed)} case(s) failed and were skipped:")
        for cid, err in failed:
            print(f"  {cid}: {err}")


if __name__ == '__main__':
    main()