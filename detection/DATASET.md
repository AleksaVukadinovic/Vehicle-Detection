# Skup podataka

Detektor se trenira na skupu **Pascal VOC 2007**, koji pored oznaka klasa
sadrži i **anotacije okvira** (bounding box) za svaki objekat — za razliku od
CIFAR-10 skupa korišćenog za referentni klasifikacioni model.

## Izvor i atribucija

- **Naziv:** The PASCAL Visual Object Classes Challenge 2007 (VOC2007)
- **Autori:** Mark Everingham, Luc Van Gool, Christopher K. I. Williams,
  John Winn, Andrew Zisserman
- **Zvanična stranica:** http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
- **Referenca:** M. Everingham i dr., *The PASCAL Visual Object Classes (VOC)
  Challenge*, International Journal of Computer Vision, 88(2), 303–338, 2010.

VOC2007 (trainval) sadrži 5.011 slika sa 20 klasa objekata; svaki objekat je
anotiran okvirom u XML formatu.

## Način pribavljanja

Skup podataka se **automatski preuzima** (~440 MB) i raspakuje u lokalni
direktorijum `data/` pri prvom pokretanju `train.py` (`src/dataset.py`,
funkcija `ensure_voc`, sa rezervnim mirror serverom). Direktorijum `data/` je
isključen iz git-a, pa je skup podataka adresiran umesto da bude smešten u
repozitorijum.

## Izbor klasa i oznaka

Od 20 klasa skupa VOC2007 koristi se 7 klasa vozila, preslikanih u jednu
klasu **„vozilo"**:

| Cilj | Originalne klase VOC2007 |
|------|--------------------------|
| vozilo | aeroplane, bicycle, boat, bus, car, motorbike, train |

Koriste se samo slike koje sadrže bar jedno vozilo (objekti označeni kao
*difficult* se preskaču), a od njih se za demo treniranje uzima podskup
(podrazumevano 500 slika, opcija `--max-images`). Preslikavanje je
implementirano u `src/config.py` (`VEHICLE_CLASSES`) i `src/dataset.py`
(`parse_vehicle_boxes`).
