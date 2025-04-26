import h5py
from pathlib import Path

def show_hdf5_tree(file_path: str | Path) -> None:
    """
    Recursively goes through all groups and datasets in an HDF5 file
    and prints them with indentation to show the hierarchy.
    """
    def _print(name, obj):
        indent = "    " * name.count("/")
        if isinstance(obj, h5py.Group):
            print(f"{indent}📂 {obj.name}/")
        else:  # Dataset
            shape = "×".join(map(str, obj.shape))
            dtype = obj.dtype
            print(f"{indent}📄 {obj.name}  (shape={shape}, dtype={dtype})")

    with h5py.File(file_path, "r") as f:
        print(f"\nInnhold i {Path(file_path).name}:")
        f.visititems(_print)

show_hdf5_tree("unseen_tables/5-heart-c_rf_mat.h5")
