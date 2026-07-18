# Skup podataka

Ovaj projekat koristi skup podataka **CIFAR-10** za treniranje i evaluaciju.

## Izvor i atribucija

- **Naziv:** CIFAR-10
- **Autori:** Alex Krizhevsky, Vinod Nair, Geoffrey Hinton
- **Zvanična stranica:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Referenca:** A. Krizhevsky, *Learning Multiple Layers of Features from Tiny
  Images*, tehnički izveštaj, Univerzitet u Torontu, 2009.

CIFAR-10 sadrži 60.000 slika u boji veličine 32×32 u 10 klasa (50.000 za
treniranje + 10.000 za testiranje), sa po 6.000 slika po klasi.

## Način pribavljanja

Skup podataka se **automatski preuzima** tokom izvršavanja pomoću biblioteke
`torchvision` (`torchvision.datasets.CIFAR10(..., download=True)`) u lokalni
direktorijum `data/` pri prvom pokretanju skripti `train.py`, `baselines.py`
ili `evaluate.py`. Direktorijum `data/` je isključen iz git-a, pa je skup
podataka *adresiran* (povezan i automatski preuzet) umesto da bude smešten u
repozitorijum.

## Preslikavanje oznaka za detekciju vozila

Originalnih 10 klasa preslikava se u binarni cilj:

| Cilj            | Originalne klase CIFAR-10              |
|-----------------|---------------------------------------|
| 1 - vozilo      | avion, automobil, brod, kamion        |
| 0 - nije vozilo | ptica, mačka, jelen, pas, žaba, konj  |

Ovo preslikavanje implementirano je u `src/config.py`
(`VEHICLE_CLASS_INDICES`).
