# Revisiting Clustering for Efficient Unsupervised Dialogue Structure Induction 
This repository contains the datasets and implementation of the Applied Intelligence, 2024 paper
["Revisiting Clustering for Efficient Unsupervised Dialogue Structure Induction"](https://link.springer.com/content/pdf/10.1007/s10489-024-05455-5.pdf) by Maarten De Raedt, Fréderic Godin, Chris Develder, and Thomas Demeester.

For any questions about the paper or code contact the first author at [maarten.deraedt@ugent.be](mailto:maarten.deraedt@ugent.be).
If you find this repository useful for your own work, consider citing our paper:

````
@article{raedt2024revisiting,
  title={Revisiting clustering for efficient unsupervised dialogue structure induction},
  author={Raedt, Maarten De and Godin, Fr{\'e}deric and Develder, Chris and Demeester, Thomas},
  journal={Applied Intelligence},
  pages={1--28},
  year={2024},
  publisher={Springer}
}
````
## Table of Contents
- [Installation](#installation)
- [Experiments](#experiment)


### Installation
Install the requirements.
```bash
$ pip3 install -r requirements.txt
```

The directory and file structure should match the structure below.
```
└─── datasets_src/
│   └──CamRest/
│   └──DSTC2/
│   └──SGD/
│   └──SimDial/
│   └──exported/
└─── datasets/
│   └──json/
│       └──camrest.json/
│       └──dstc2.json/
│       └──sgd-events_2.json/
│       └──sgd-homes_1.json/
│       └──sgd-movies_1.json/
│       └──sgd-music_2.json/
│       └──simdial-bus.json/
│       └──simdial-movie.json/
│       └──simdial-restaurant.json/
│       └──simdial-weather.json/
│   └──tojson.py
│   └──camrest-all-MiniLM-L6-v2.pkl
│   └──dst2-all-MiniLM-L6-v2.pkl
│   └──sgd-*-all-MiniLM-L6-v2.pkl
│   └──simdial-*-all-MiniLM-L6-v2.pkl
│   dataloaders.py
│   dialogs.sh
│   embedders.py
│   main.py
│   main_turn.py
│   models.py
│   README.md
│   requirements.txt
```

### Experiments
The `datasets/` directory contains the pickled datasets (along with a json export) with the utterances embedded using the MiniLM encoder.
The `datasets-src/`directory contains the datasets in their unmodified form, i.e., as originally released.
The experiments use the exported, pickled datasets in the `datasets/` directory.

To reproduce the main results with turn based evaluation and MiniLM:
```bash
python3 main_turn.py
```
and for the results with utterance based evaluation for MiniLM: 
```bash
python3 main.py
```
The results will be written to a .json file in the corresponding directory in `results/`. 
To experiment with a sentence encoder other than MiniLM (e.g., GloVe), you first have to set the encoder in embedder.py, followed by exporting the datasets of interest.
