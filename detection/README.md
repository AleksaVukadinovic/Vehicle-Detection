# Detekcija vozila sa lokalizacijom (jednostepeni detektor)

Jednostepeni (single-stage) detektor objekata u stilu SSD/YOLO arhitektura,
izgrađen od nule u PyTorch-u. Za datu ulaznu sliku model vraća kopiju slike sa
**uokvirenim vozilima** — svaki pronađeni objekat dobija pravougaonik i ocenu
pouzdanosti.

Model se trenira na skupu [Pascal VOC 2007](DATASET.md) (klase vozila), koji se
automatski preuzima pri prvom pokretanju.

> **Napomena o obimu treniranja:** cilj projekta je demonstracija arhitekture i
> tehnike detekcije objekata, a ne vrhunske performanse. Podrazumevano se
> trenira kratak demo (podskup od 1.500 slika, 30 epoha, nekoliko minuta na
> GPU/MPS), pa su detekcije orijentacione, a ocene pouzdanosti niske
> (podrazumevani prag je 0,3). Za bolje rezultate povećati `--max-images` i
> `--epochs`.

## Arhitektura

Detektor (`src/model.py`) prati standardnu jednostepenu šemu sa anchor
kutijama:

1. **Backbone** — 4 konvoluciona bloka (2× 3×3 konv → BatchNorm → ReLU, prva
   tri sa MaxPool), progresija kanala 3 → 32 → 64 → 128 → 256. Ulaz 128×128
   se svodi na mapu odlika 16×16 (korak 8 piksela).
2. **Detekciona glava** — zajednički 3×3 konvolucioni sloj, pa dve paralelne
   1×1 konvolucije:
   - *objectness* grana: po jedna ocena za svaku anchor kutiju (da li kutija
     sadrži vozilo),
   - *regresiona* grana: 4 pomeraja (Δx, Δy, Δw, Δh) kojima se anchor kutija
     deformiše u tačan okvir objekta.
3. **Anchor kutije** (`src/anchors.py`) — u svakoj od 16×16 ćelija mreže
   generiše se 9 kutija (3 skale × 3 odnosa stranica), ukupno 2304 kandidata
   po slici.

## Tehnika treniranja

- **Uparivanje** (`src/anchors.py`): anchor kutija je pozitivna ako joj je
  IoU sa nekim stvarnim okvirom ≥ 0,5 (najbolja kutija za svaki objekat je
  uvek pozitivna), negativna ako je IoU < 0,4, a između se ignoriše.
- **Funkcija gubitka** (`src/loss.py`): binarna unakrsna entropija za
  objectness sa *hard negative mining* (odnos negativnih i pozitivnih 3:1,
  kao kod SSD-a) + smooth L1 za regresiju pomeraja na pozitivnim kutijama.
- **Inferencija** (`detect.py`): sigmoid ocene → prag pouzdanosti →
  dekodiranje pomeraja u koordinate → *non-maximum suppression* (NMS) za
  uklanjanje preklapajućih detekcija → crtanje okvira na originalnoj slici.

## Treniranje

```bash
python train.py                        # demo: 1500 slika, 30 epoha
python train.py --epochs 60 --max-images 5000 --lr 1e-3
```

Dostupne opcije: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`,
`--max-images`, `--num-workers`, `--seed`.

VOC2007 se automatski preuzima u `data/` pri prvom pokretanju. Najbolji model
(po validacionom gubitku) čuva se u `checkpoints/best_detector.pt`, a dnevnik
treniranja u `checkpoints/history.json`.

## Inferencija — uokvirivanje vozila

```bash
python detect.py --image ulica.jpg
python detect.py --image a.jpg b.jpg --score-threshold 0.4
```

Za svaku ulaznu sliku snima se kopija sa nacrtanim okvirima u
`outputs/<ime>_detected.<ekstenzija>` i ispisuju koordinate detekcija:

```
ulica.jpg: 2 vehicle(s) detected -> outputs/ulica_detected.jpg
  box=(135, 129, 421, 263) score=31.84%
  box=(220, 211, 343, 263) score=30.60%
```

## Skup podataka

Pogledajte [`DATASET.md`](DATASET.md) za izvor skupa podataka, atribuciju i
izbor klasa vozila.
