#!/usr/bin/env python3
"""
Visualize displacement fields from RAFT output .mat files.

Usage:
    python scripts/visualize.py --mat outputs/raft_disp/envelope_flow41__envelope_flow42_flow.mat
    python scripts/visualize.py --mat outputs/raft_disp/raft_disp_all.mat --pair_idx 0
    python scripts/visualize.py --mat outputs/raft_disp/raft_disp_all.mat --pair_idx 0 --stride 20
"""
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description="Visualize displacement fields from RAFT output .mat files"
    )
    ap.add_argument("--mat", required=True, help="Path to *_flow.mat with disp_x/disp_y")
    ap.add_argument("--stride", type=int, default=10, help="Quiver stride (default: 10)")
    ap.add_argument("--pair_idx", type=int, default=None, 
                    help="For single_mat files, which pair index to visualize (default: 0)")
    ap.add_argument("--save", type=str, default=None,
                    help="Save figure to file instead of showing (e.g., 'output.png')")
    args = ap.parse_args()

    mat_path = Path(args.mat)
    if not mat_path.exists():
        print(f"Error: File not found: {mat_path}")
        return

    print(f"Loading: {mat_path}")
    m = sio.loadmat(str(mat_path))
    
    # Check if this is a single_mat file (has 'names' key) or per_pair file
    if "names" in m:
        # single_mat format: disp_x and disp_y are [N, H, W]
        disp_x_all = np.asarray(m["disp_x"])
        disp_y_all = np.asarray(m["disp_y"])
        names = m["names"]
        
        if disp_x_all.ndim == 3:
            # Multiple pairs
            pair_idx = args.pair_idx if args.pair_idx is not None else 0
            if pair_idx >= disp_x_all.shape[0]:
                print(f"Error: pair_idx {pair_idx} out of range (max: {disp_x_all.shape[0]-1})")
                return
            
            disp_x = disp_x_all[pair_idx]
            disp_y = disp_y_all[pair_idx]
            pair_name = names[pair_idx][0] if isinstance(names[pair_idx], np.ndarray) else str(names[pair_idx])
            print(f"Visualizing pair {pair_idx}: {pair_name}")
        else:
            # Single pair in single_mat format
            disp_x = disp_x_all
            disp_y = disp_y_all
            pair_name = "pair_0"
    else:
        # per_pair format: disp_x and disp_y are [H, W]
        disp_x = np.asarray(m["disp_x"])
        disp_y = np.asarray(m["disp_y"])
        pair_name = mat_path.stem

    if disp_x.shape != disp_y.shape:
        print(f"Error: disp_x shape {disp_x.shape} != disp_y shape {disp_y.shape}")
        return

    mag = np.sqrt(disp_x**2 + disp_y**2)

    # Figure 1: Displacement magnitude
    plt.figure(figsize=(10, 8))
    plt.title(f"Displacement magnitude - {pair_name}")
    im = plt.imshow(mag, aspect="auto", cmap="hot")
    plt.colorbar(im, label="Magnitude")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    if args.save:
        save_path = Path(args.save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        plt.savefig(save_path.with_stem(save_path.stem + "_magnitude"), dpi=150, bbox_inches="tight")
        print(f"Saved magnitude plot: {save_path.with_stem(save_path.stem + '_magnitude')}")
    else:
        plt.show(block=False)

    # Figure 2: Displacement vectors (quiver)
    s = args.stride
    yy, xx = np.mgrid[0:disp_x.shape[0]:s, 0:disp_x.shape[1]:s]
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Displacement vectors (stride={s}) - {pair_name}")
    plt.imshow(mag, aspect="auto", cmap="gray", alpha=0.5)
    plt.quiver(xx, yy, disp_x[::s, ::s], disp_y[::s, ::s], 
               angles="xy", scale_units="xy", scale=1, 
               color="red", width=0.003)
    plt.xlabel("X")
    plt.ylabel("Y")
    
    if args.save:
        save_path = Path(args.save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        plt.savefig(save_path.with_stem(save_path.stem + "_vectors"), dpi=150, bbox_inches="tight")
        print(f"Saved vectors plot: {save_path.with_stem(save_path.stem + '_vectors')}")
    else:
        plt.show()

    if not args.save:
        print("\nClose figures to exit.")


if __name__ == "__main__":
    main()

