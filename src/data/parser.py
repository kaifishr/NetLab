"""Script holds methods to parse data from Tensorboard event files.
"""

import glob
import os
import traceback

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tensorboard_event_to_pandas(path: str) -> pd.DataFrame:
    """Convert single tensorflow log file to pandas DataFrame."""
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 0,
        "images": 0,
        "scalars": 0,  # 0 all events are stored
        "histograms": 0,
    }
    # TENSORS = "tensors"
    # GRAPH = "graph"
    # META_GRAPH = "meta_graph"
    # RUN_METADATA = "run_metadata"
    # COMPRESSED_HISTOGRAMS = "distributions"
    # HISTOGRAMS = "histograms"
    # IMAGES = "images"
    # AUDIO = "audio"
    # SCALARS = "scalars"

    events = dict()
    try:
        event_acc = EventAccumulator(path=path, size_guidance=DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        events = {tag: dict() for tag in tags}
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            events[tag]["step"] = list(map(lambda x: x.step, event_list))
            events[tag]["value"] = list(map(lambda x: x.value, event_list))
            events[tag]["wall_time"] = list(map(lambda x: x.wall_time, event_list))
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()

    return pd.DataFrame.from_dict(events)


def main(log_dir: str, write_pkl: bool, write_csv: bool, out_dir: str):
    """ """
    if os.path.isdir(log_dir):
        event_paths = glob.glob(os.path.join(log_dir, "*", "event*"))
    elif os.path.isfile(log_dir):
        event_paths = [log_dir]
    else:
        raise ValueError(f"{log_dir} has to be a file or directory.")

    # Call & append
    if event_paths:

        os.makedirs(out_dir, exist_ok=True)

        for event_path in event_paths:
            events = tensorboard_event_to_pandas(path=event_path)
            file_name = f"{event_path.split('/')[-2]}"
            out_file = os.path.join(out_dir, file_name)
            events.to_pickle(f"{out_file}.pkl")
            # events.to_csv(f"{out_file}.csv")
    else:
        print("No event paths have been found.")


if __name__ == "__main__":
    log_dir = "../runs"
    out_dir = "../results"
    main(log_dir=log_dir, write_pkl=False, write_csv=False, out_dir=out_dir)
