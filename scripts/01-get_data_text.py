import os
import subset2evaluate.utils
import csv
import sklearn.model_selection

os.makedirs("data", exist_ok=True)

data_all = [
    {
        "src": x["src"],
        "mt": x["tgt"][sys],
        "score": x["scores"][sys]["human"] if x["scores"][sys]["human"] >= 1 else 100*x["scores"][sys]["human"],
        "src_audio": "",
    }
    for data in (subset2evaluate.utils.load_data_wmt_all() | subset2evaluate.utils.load_data_biomqm()).values()
    for x in data
    if "speech" not in x["domain"]
    for sys in x["tgt"]
]

data_train, data_dev = sklearn.model_selection.train_test_split(data_all, test_size=1000)

with open("data/text_train.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["score", "src", "mt", "src_audio"])
    writer.writeheader()
    writer.writerows(data_train)

with open("data/text_dev.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["score", "src", "mt", "src_audio"])
    writer.writeheader()
    writer.writerows(data_dev)