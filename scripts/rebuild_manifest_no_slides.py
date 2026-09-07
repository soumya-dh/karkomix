import pandas as pd
import sys

all_files = pd.read_csv(sys.argv[1], sep="\t")   # all_files_by_modality.tsv
selected = pd.read_csv(sys.argv[2], sep="\t")     # selected_cases.tsv
out_path = sys.argv[3]                            # new manifest.txt path

selected_ids = set(selected["case_submitter_id"])
manifest_files = all_files[
    (all_files["case_submitter_id"].isin(selected_ids)) &
    (all_files["modality"] != "slide_image")
]

manifest = manifest_files[["file_id"]].drop_duplicates()
manifest.columns = ["id"]
manifest.to_csv(out_path, sep="\t", index=False)

print(f"Original files (with slides): {len(all_files[all_files['case_submitter_id'].isin(selected_ids)])}")
print(f"New manifest (no slides): {len(manifest)} files -> {out_path}")
print()
print("Modality breakdown in new manifest:")
print(manifest_files["modality"].value_counts())
