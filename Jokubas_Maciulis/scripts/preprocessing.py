import pandas as pd
import numpy as np
import awkward as ak
import uproot

def load_dataset_from_txt(txt_file, target_label, max_events = None, branches = None):
    
    arrays = []
    loaded = 0

    files = np.loadtxt(txt_file, dtype=str)

    for i, file_path in enumerate(files):
        if max_events is not None and loaded >= max_events:
            break

        with uproot.open(file_path) as root_file:
            tree = root_file["Events"]

            events_left = None
            if max_events is not None:
                events_left = max_events - loaded

            arr = tree.arrays(
                branches, 
                library="ak",
                entry_stop=events_left
            )

        arrays.append(arr)
        loaded += len(arr)

    data = ak.concatenate(arrays)
    df = ak.to_dataframe(data)
    df["target"] = target_label
    # data = ak.with_field(data, np.full(len(data), target_label), "target")
    return df.reset_index(drop=True)

branches = ["Electron_pt", "Electron_eta", "run", "event"]
print(load_dataset_from_txt("data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_270000_file_index.txt", 1, 3, branches = branches))