# Detekcija vozila (PyTorch)

Projekat rešava problem **detekcije vozila na slici**: za datu ulaznu sliku
model vraća istu sliku sa **uokvirenim vozilima** (pravougaonici oko svakog
detektovanog vozila). Repozitorijum sadrži dva modela u odvojenim
direktorijumima:

| Direktorijum | Model | Zadatak | Skup podataka |
|--------------|-------|---------|---------------|
| [`detection/`](detection/) | **Jednostepeni detektor objekata** (glavni model) | Lokalizacija: vraća sliku sa uokvirenim vozilima | Pascal VOC 2007 |
| [`classification/`](classification/) | KNM binarni klasifikator (referentni model) | Klasifikacija: vozilo / nije vozilo, bez lokalizacije | CIFAR-10 |

Klasifikacioni model je zadržan kao dodatni referentni (baseline) model za
poređenje — pokazuje šta konvoluciona mreža može bez detekcione glave, dok
glavni model u `detection/` rešava zadatak lokalizacije okvira.

## Podešavanje okruženja

Okruženje je zajedničko za oba modela i kreira se u korenu repozitorijuma:

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Brzi početak — detekcija vozila

```bash
cd detection
python train.py                       # kratko demo treniranje (VOC2007 se sam preuzima)
python detect.py --image ulica.jpg    # snima outputs/ulica_detected.jpg sa okvirima
```

Detalji o arhitekturi detektora, funkciji gubitka i skupu podataka nalaze se u
[`detection/README.md`](detection/README.md).

## Referentni klasifikacioni model

```bash
cd classification
python train.py --epochs 20
python predict.py --image auto.jpg
```

Detalji u [`classification/README.md`](classification/README.md).

## Dokumentacija i prezentacija

Naučni izveštaj i prezentacija za odbranu nalaze se u direktorijumu `docs/`.
