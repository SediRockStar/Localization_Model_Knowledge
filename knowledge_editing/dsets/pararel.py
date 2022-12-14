import json
import typing
from pathlib import Path
from tqdm import tqdm
import collections
import torch
from torch.utils.data import Dataset
import urllib.request

from util.globals import *

REMOTE_URL = f"https://raw.githubusercontent.com/yanaiela/pararel/main/data/pattern_data/graphs_json/.jsonl"

PARAREL_RELATION_NAMES = [
    "P39",
    "P264",  # Remove the last three patterns
    "P108",
    "P131",
    "P176",  # Remove the last two patterns
    "P30",
    "P178",  # Remove the last four patterns
    "P138",  # Remove pattern 11, 12, 13
    "P47",  # Remove last 4 relations / total 9
    "P27",  # Remove the first one and the last pattern
    "P364",  # Remove last two
    "P495",
    "P449",  # Remove 4 out of 11
    "P20",
    "P36",  # Remove 6 out of 14
    "P19",  # Remove 6 out of 13
    "P740",
    "P279",
    "P159",  # Remove 3 out of 10
    "P106",  # Remove first 4 out of 11
    "P101",  # Remove 3 out of 11
    "P937",
]


class PaRaRelDataset(Dataset):
    def __init__(
            self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        data_dir = Path(data_dir)
        para_loc = data_dir / "para.json"
        if not para_loc.exists():
            print(f"{para_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            PARAREL = collections.defaultdict(dict)
            # download relations from github
            for r in tqdm(PARAREL_RELATION_NAMES, "downloading pararel data"):
                with urllib.request.urlopen(
                        f"https://raw.githubusercontent.com/yanaiela/pararel/main/data/pattern_data/graphs_json/{r}.jsonl"
                ) as url:
                    graphs = [
                        json.loads(d.strip()) for d in url.read().decode().split("\n") if d
                    ]
                    # Prune the punctuations
                    graphs = [
                        {'pattern': graph['pattern'].strip('.').strip(',').strip(), 'lemma': graph['lemma'],
                         'extended_lemma': graph['extended_lemma'], 'tense': graph['tense']} for graph in graphs
                    ]

                    # Remove the patterns where [Y] is not at the end of the sentence
                    # by manually check if [Y] is located at the end of the sentence
                    graphs = [
                        {'pattern': graph['pattern'], 'lemma': graph['lemma'],
                         'extended_lemma': graph['extended_lemma'], 'tense': graph['tense']} for graph in graphs
                        if (graph['pattern'].rindex("[Y]") + 3 == len(graph['pattern']) or graph['pattern'].rindex(
                            "[Y]") + 4 == len(graph['pattern']))  #
                    ]

                    # Manually add the punctuations back for the models for convienient in-context learning
                    graphs = [
                        {'pattern': graph['pattern'] if graph['pattern'][-1] in '?.,!;' else graph['pattern'] + '.',
                         'lemma': graph['lemma'],
                         'extended_lemma': graph['extended_lemma'], 'tense': graph['tense']} for graph in graphs
                    ]

                    PARAREL[r]["graphs"] = graphs
                with urllib.request.urlopen(
                        f"https://raw.githubusercontent.com/yanaiela/pararel/main/data/trex_lms_vocab/{r}.jsonl"
                ) as url:
                    vocab = [
                        json.loads(d.strip()) for d in url.read().decode().split("\n") if d
                    ]
                    PARAREL[r]["vocab"] = vocab
            with open(para_loc, "w") as f:
                json.dump(PARAREL, f)

        with open(para_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
