# Klasifikacija vozila pomoću KNM (referentni model)

Konvoluciona neuronska mreža (KNM) za **binarnu klasifikaciju** slika -
*vozilo* naspram *nije vozilo* - napravljena pomoću PyTorch-a. Trenira se na
skupu podataka [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
(60.000 slika), koji se automatski preuzima pri prvom pokretanju.

Ovaj model služi kao **referentni (baseline) model** za poređenje: klasifikuje
celu sliku, ali ne lokalizuje objekte. Glavni model projekta, koji vraća sliku
sa uokvirenim vozilima, nalazi se u direktorijumu
[`../detection/`](../detection/).

## Arhitektura

Kompaktna KNM u stilu VGG (`src/model.py`):

- 3 konvoluciona bloka (svaki: 2× 3×3 konv → BatchNorm → ReLU → MaxPool)
- Progresija kanala 3 → 64 → 128 → 256
- Globalno usrednjeno objedinjavanje + klasifikaciona glava sa 2 sloja i dropout-om
- Kaiming inicijalizacija težina

Mreža se automatski izvršava na **CUDA**, **Apple MPS** ili **procesoru (CPU)**
(izbor u `src/config.py`).

## Rezultati

Performanse na test skupu (10.000 slika) za zadatak binarne klasifikacije vozila,
poređenje KNM sa dve samostalno razvijene referentne metode na sirovim pikselima:

| Model | Tačnost | Preciznost | Odziv | F1 | ROC-AUC |
|-------|:-------:|:----------:|:-----:|:--:|:-------:|
| Logistička regresija | 0,816 | 0,791 | 0,733 | 0,761 | 0,879 |
| MLP (bez konvolucija) | 0,880 | 0,854 | 0,845 | 0,850 | 0,943 |
| **VehicleCNN (naše)** | **0,973** | **0,969** | **0,963** | **0,966** | **0,997** |

Konvoluciona arhitektura je najbolja: poboljšava tačnost za ~9
procentnih poena u odnosu na nelinearni MLP uporedive složenosti.

## Podešavanje okruženja

Okruženje je zajedničko za ceo repozitorijum i kreira se u korenu (pogledajte
[`../README.md`](../README.md)):

```bash
cd ..
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
cd classification
```

## Treniranje

```bash
python train.py --epochs 20 --batch-size 128 --lr 1e-3
```

Dostupne opcije: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`,
`--num-workers`, `--seed`.

Tokom treniranja skup podataka se automatski preuzima u `data/`. Najbolji model
(po validacionoj tačnosti) čuva se u `checkpoints/best_model.pt`, a dnevnik
treniranja u `checkpoints/history.json`. Na kraju se ispisuje završna evaluacija
na izdvojenom test skupu.

## Inferencija

Klasifikacija sopstvenih slika kao vozilo / nije vozilo:

```bash
python predict.py --image putanja/do/slike.jpg
python predict.py --image auto.jpg pas.jpg --checkpoint checkpoints/best_model.pt
```

Primer izlaza:

```
auto.jpg: VEHICLE (vehicle probability = 98.42%)
pas.jpg: NOT A VEHICLE (vehicle probability = 3.11%)
```

## Reprodukcija eksperimenata i slika

```bash
python train.py --epochs 20 --num-workers 0   # treniranje KNM
python baselines.py                            # treniranje logističke regresije + MLP
python evaluate.py                             # generisanje slika + tabele poređenja
```

Ovo popunjava `reports/figures/` (krive treniranja, matrica konfuzije, ROC/PR
krive, primeri predviđanja) i `reports/comparison.json`.

## Dokumentacija i prezentacija

Naučni izveštaj i prezentacija za odbranu nalaze se u direktorijumu
[`../docs/`](../docs/). Dokumenti uključuju slike iz `reports/figures/`, pa ih
pre ažuriranja regenerišite pomoću `evaluate.py`.

## Skup podataka

Pogledajte [`DATASET.md`](DATASET.md) za izvor skupa podataka, atribuciju i
binarno preslikavanje oznaka. Skup podataka automatski preuzima torchvision.

## Napomene

- Direktorijumi `data/` i `checkpoints/` su isključeni iz git-a (`.gitignore`).
- Za najbolju brzinu koristite CUDA GPU; Apple Silicon automatski koristi MPS.
- Na CPU/MPS, postavite `--num-workers 0` ako naiđete na probleme pri treniranju
