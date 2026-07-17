# Priprema za odbranu — Detekcija vozila pomoću KNM (CNN)

Ovaj dokument je detaljna analiza projekta, napisana tako da možeš da objasniš
**svaku** odluku u kodu i, što je najvažnije, **zašto** je baš tako urađena.
Na kraju se nalazi lista pitanja koja bi profesor/asistent najverovatnije
postavio, sa spremnim odgovorima.

---

## 1. Cilj projekta i postavka problema

Projekat rešava **binarnu klasifikaciju slika**: da li slika sadrži vozilo ili
ne. Koristi se javno dostupan skup podataka **CIFAR-10** (60.000 slika u boji,
32×32 piksela, 10 klasa, po 6.000 slika po klasi — 5.000 trening + 1.000 test
po klasi).

Originalnih 10 klasa preslikano je u binarni cilj:

| Cilj | CIFAR-10 klase | Broj klasa |
|------|-----------------|:---:|
| **1 — vozilo** | avion, automobil, brod, kamion | 4 |
| **0 — nije vozilo** | ptica, mačka, jelen, pas, žaba, konj | 6 |

**Zašto baš ove četiri klase kao "vozilo"?** Avion, automobil, brod i kamion
su sva sredstva za prevoz (transportna sredstva) — logička definicija pojma
"vozilo" u širem smislu. Ostalih šest klasa su životinje, dakle jasno
"ne-vozilo". Podela je nedvosmislena, nema graničnih/dvosmislenih slučajeva,
što je bitno jer nema mogućnosti za "prljave" oznake (label noise) usled same
definicije problema.

**Zašto je bitno primetiti da skup NIJE savršeno balansiran?** 4 klase vozila
= 40% podataka, 6 klasa ne-vozila = 60% podataka. Ovo je blaga (ne teška)
neuravnoteženost, ali je dovoljan razlog da se pored *accuracy* prate i
precision/recall/F1/ROC-AUC (videti sekciju 7) — accuracy sam po sebi može biti
zavaravajući kad klase nisu 50/50.

Ovo je implementirano u `src/config.py` (`VEHICLE_CLASS_INDICES = {0, 1, 8, 9}`)
i primenjeno u `src/dataset.py` (`VehicleBinaryDataset.__getitem__`), gde se
originalna oznaka (0–9) mapira u binarnu (0/1) u trenutku učitavanja svakog
uzorka.

---

## 2. Skup podataka i pretprocesiranje

### 2.1. Podela na train / validation / test

- **Test skup** (10.000 slika) je zaseban, standardni CIFAR-10 test split — nikad
  se ne koristi za treniranje niti za izbor hiperparametara.
- **Trening skup** (50.000 slika) se dalje deli na **90% trening / 10%
  validacija** (`val_split = 0.1` u `TrainConfig`), koristeći
  `torch.utils.data.random_split` sa fiksnim seed-om (`seed=42`).

**Zašto uopšte treba validacioni skup, zar test skup nije dovoljan?**
Validacioni skup se koristi da se u toku treniranja izabere *koji je epoch dao
najbolji model* (`best_model.pt` se čuva na osnovu najveće `val_acc`). Kad bismo
za tu odluku koristili test skup, test rezultat više ne bi bio nepristrasna
(unbiased) procena generalizacije — praktično bismo "petljali" test skup u
proces treniranja. Zato test skup ostaje potpuno nedodirnut do samog kraja,
gde se koristi tačno jednom, za finalni izveštaj.

**Zašto fiksni seed (42)?** Radi reproducibilnosti — da se podela na
train/val i inicijalizacija težina mogu ponoviti i da rezultati nisu
"srećni slučaj" jedne nasumične podele.

**Bitan detalj:** validacioni podskup preuzima indekse iz `random_split`, ali
mu se ručno menja `dataset` atribut da koristi **eval transformacije** (bez
augmentacije) — `val_subset.dataset = val_base` u `src/dataset.py`. Ovo je
namerno: da augmentacija (nasumični crop/flip) ne "zamuti" validacionu ocenu,
jer validacija treba da simulira uslove iz stvarne primene (test/inferenciju),
gde nema augmentacije.

### 2.2. Normalizacija piksela

```python
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

Ovo su standardne, **unapred izračunate statistike CIFAR-10 skupa** (srednja
vrednost i standardna devijacija piksela po RGB kanalima, na celom trening
skupu, u opsegu [0,1]).

**Zašto normalizacija uopšte?** Ulazni pikseli su u opsegu [0, 255] (odnosno
[0,1] posle `ToTensor()`). Bez centriranja i skaliranja, ulazne vrednosti bi
imale veliku, nekonzistentnu skalu što otežava optimizaciju: gradijenti prvog
sloja bi zavisili od apsolutnih vrednosti piksela, konvergencija bi bila
sporija i osetljivija na learning rate. Normalizacija na priblizno
nula-srednju-vrednost/jediničnu-varijansu ulaz čini brojno stabilnijim za
gradijentni spust i za rad Batch Normalization slojeva unutar mreže.

**Zašto se koriste baš statistike *CIFAR-10 skupa* a ne, recimo, ImageNet
statistike?** Zato što mreža uči *od nule* (nije predtrenirana / transfer
learning), pa treba da normalizacija odgovara distribuciji podataka na kojima
se zaista trenira — ne postoji razlog da se koristi statistika nekog drugog
skupa.

### 2.3. Augmentacija podataka (samo za trening)

```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
```

**Zašto augmentacija?** Sa 1,18M parametara i samo ~45.000 trening slika,
mreža ima kapacitet da nauči i zapamti "napamet" (overfituje) originalne
slike. Augmentacija veštački uvećava raznovrsnost trening skupa tako što na
svaku epohu daje malo drugačiju verziju iste slike, što deluje kao regularizacija
i sprečava mrežu da memoriše tačne piksele umesto da uči generalizabilne
oblike/teksture.

- **RandomCrop(32, padding=4):** slika se prvo popuni (padding) sa po 4 piksela
  crne/nulte granice sa svake strane (32→40×40), pa se nasumično iseče
  32×32 isečak. Ovo simulira male pomeraje objekta u kadru (translational
  invariance) — model uči da prepozna vozilo bez obzira gde se tačno nalazi
  u okviru slike.
- **RandomHorizontalFlip():** slika se sa verovatnoćom 50% horizontalno
  izvrne. Ima smisla za vozila i životinje jer njihov identitet ne zavisi od
  toga da li gledaju levo ili desno (horizontalna simetrija scene). *Vertikalni*
  flip namerno nije korišćen jer bi proizveo nerealne slike (auto naopako,
  naglavačke životinje se ne pojavljuju u stvarnim fotografijama).

**Zašto NE i color jitter / rotacije / veći uglovi?** Za male 32×32 slike (već
niska rezolucija), agresivnija augmentacija (jako zasićenje boja, rotacije od
recimo 30°) lako uništi već ograničenu informaciju u slici i može da škodi
umesto da pomogne. Blaga augmentacija (crop + flip) je standardna, dokazana
praksa za CIFAR-10 iz literature i dovoljna je za ovaj obim mreže.

**Zašto se augmentacija NE primenjuje na validacioni i test skup?**
Augmentacija je alatka isključivo za regularizaciju u treningu. Validacija i
test moraju da mere performansu na *realnim*, nepromenjenim slikama — jer
tako će model biti korišćen u praksi (`predict.py` takođe ne augmentira ulaznu
sliku, samo je skalira i normalizuje).

---

## 3. Arhitektura mreže — `VehicleCNN`

Mreža je kompaktna **konvoluciona neuronska mreža u VGG stilu**
(naizmenični parovi konvolucija + pooling, sve do globalnog pooling-a i
potpuno povezanog klasifikatora).

```
Ulaz: 3×32×32 (RGB)
 └─ ConvBlock(3→64)    : [Conv3x3→BN→ReLU] × 2 → MaxPool2   → 64×16×16
 └─ ConvBlock(64→128)  : [Conv3x3→BN→ReLU] × 2 → MaxPool2   → 128×8×8
 └─ ConvBlock(128→256) : [Conv3x3→BN→ReLU] × 2 → MaxPool2   → 256×4×4
 └─ AdaptiveAvgPool2d(1)                                     → 256×1×1
 └─ Flatten                                                  → 256
 └─ Dropout(0.5) → Linear(256→128) → ReLU
 └─ Dropout(0.5) → Linear(128→2)                             → logiti (2 klase)
```

Ukupno **1.180.354 parametara** (dobijeno direktnim brojanjem iz modela).

### 3.1. Zašto konvolucije umesto potpuno povezane (FC) mreže?

Ovo je i eksperimentalno dokazano u samom projektu (`baselines.py` /
`evaluate.py`): MLP bez konvolucija na istim sirovim pikselima dostiže
88,0% tačnosti, dok CNN dostiže 97,3%. Razlog je fundamentalan:

- **Lokalna povezanost (locality):** konvolucija posmatra mali lokalni prozor
  (3×3) — piksel je najviše povezan sa svojim susedima, što odgovara prirodi
  slika (ivice, teksture su lokalne pojave). FC sloj bi tretirao svaki piksel
  kao nezavisnu, ravnopravnu ulaznu promenljivu, gubeći prostornu strukturu.
- **Deljenje težina (weight sharing):** isti filter (kernel) klizi po celoj
  slici, pa mreža uči da prepozna obrazac (npr. ivicu ili teksturu točka)
  *bez obzira gde se on nalazi na slici* — što drastično smanjuje broj
  parametara u odnosu na FC sloj iste "moći zapažanja" i poboljšava
  generalizaciju.
- **Translaciona invarijantnost:** posledica deljenja težina — pomeranje
  objekta u kadru ne menja suštinski predstavu koju mreža izvlači.

### 3.2. `ConvBlock` — zašto baš ova struktura (2× konv → BN → ReLU → pool)?

**Zašto 3×3 kernel?** Ovo je VGG princip (Simonyan & Zisserman, 2014):
dva uzastopna 3×3 konvoluciona sloja imaju isto **receptivno polje** kao jedan
5×5 sloj (5×5 = jedan piksel "vidi" oblast 5×5 iz prethodnog sloja), ali sa
**manje parametara** (2×(3×3)=18 vs 1×(5×5)=25 po ulazno-izlaznom paru kanala)
i, bitnije, sa **dve nelinearnosti (ReLU) umesto jedne** — mreža tako uči
složeniju, dublju nelinearnu transformaciju uz manju cenu.

**Zašto `padding=1` uz `kernel_size=3`?** Padding=1 čuva prostornu dimenziju
nepromenjenom (`"same"` padding: ulaz 32×32 → izlaz 32×32). Ovo omogućava da
se dubina mreže (broj slojeva) bira nezavisno od gubitka rezolucije usled
same konvolucije — rezolucija se smanjuje isključivo kontrolisano, kroz
`MaxPool2d`, a ne slučajno kroz nedostatak paddinga.

**Zašto Batch Normalization posle svake konvolucije (a pre ReLU)?**
- Normalizuje aktivacije (srednja vrednost ≈0, varijansa ≈1) unutar svakog
  mini-batch-a, što stabilizuje i ubrzava konvergenciju — mreža može da
  koristi veći learning rate bez divergencije.
- Blago regularizuje trening (statistika batch-a unosi malo "šuma"), što
  dodatno pomaže generalizaciji.
- Postavlja se pre ReLU jer se centrira raspodela oko nule pre nelinearnog
  odsecanja negativnih vrednosti — ovo je standardan i eksperimentalno
  potvrđen redosled iz originalnog BatchNorm rada (Ioffe & Szegedy, 2015).

**Zašto ReLU?** Jednostavna, računski jeftina nelinearnost
(`max(0, x)`), koja ne pati od problema nestajućeg gradijenta (vanishing
gradient) kao sigmoid/tanh za pozitivne ulaze, i empirijski najbolje radi u
konvolucionim mrežama.

**Zašto MaxPool2d(2) na kraju svakog bloka, a ne stride=2 u konvoluciji ili
avg pooling?**
- MaxPool prepolovljuje prostorne dimenzije (32→16→8→4) i time: (1) smanjuje
  računsku cenu sledećih slojeva, (2) uvećava receptivno polje dubljih slojeva
  relativno na ulaznu sliku, (3) unosi malu translacionu invarijantnost na
  lokalnom nivou (mala pomeranja unutar 2×2 prozora ne menjaju izlaz).
- Max (a ne Average) pooling se bira jer čuva *najizraženiju* aktivaciju u
  prozoru — u zadacima detekcije objekata to obično odgovara najjačem odzivu
  na neku vizuelnu karakteristiku (ivicu, deo oblika), što je informativnije
  od "razvodnjavanja" prosekom.

**Zašto progresija kanala 3 → 64 → 128 → 256 (duplira se svaki blok)?**
Kako se prostorna rezolucija smanjuje (manje piksela po mapi osobina), broj
kanala (dubina) se povećava kako bi mreža nadoknadila gubitak prostorne
informacije većim brojem naučenih apstraktnih obrazaca po lokaciji. Ovo je
uobičajen dizajn (i u VGG-u i u većini modernih CNN): plitki slojevi uče
jednostavne osobine (ivice, boje, teksture) sa malo kanala, duboki slojevi
uče apstraktnije, semantičke osobine (delovi objekta, oblici) i zato im treba
veći kapacitet (više kanala).

### 3.3. Zašto Global Average Pooling (`AdaptiveAvgPool2d(1)`) umesto Flatten + veliki FC sloj?

Posle tri bloka mapa osobina je oblika `256×4×4`. Klasičan pristup
(npr. originalni AlexNet/VGG) bio bi da se ovo spljošti (`flatten`) u vektor
od `256×4×4 = 4096` i poveže sa velikim FC slojem — što uvodi ogroman broj
dodatnih parametara (npr. `4096×128 ≈ 524.000` samo za prvi FC sloj) i time
veći rizik od overfitting-a.

Global Average Pooling umesto toga uzima *prosek* svake od 256 mapa osobina
preko celog 4×4 prostora, dajući vektor dužine samo 256:

- **Drastično manje parametara** u klasifikatoru → manji rizik od
  overfitting-a, brže treniranje.
- **Prostorna nezavisnost od veličine ulaza:** `AdaptiveAvgPool2d(1)` radi za
  bilo koju ulaznu rezoluciju (ne samo tačno 32×32) — mreža bi mogla da
  primi i veće slike bez menjanja arhitekture (mada projekat u praksi uvek
  skalira ulaz na 32×32 u `predict.py`, zbog konzistentnosti sa treningom).
- Ovaj pristup je popularizovan u Network-in-Network / GoogLeNet radovima
  upravo kao zamena za glomazne FC slojeve na kraju CNN-a.

### 3.4. Klasifikaciona glava — zašto Dropout, zašto dve FC transformacije?

```
Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→2)
```

**Zašto Dropout(0.5)?** Tokom treninga, nasumično se "gasi" (postavlja na 0)
50% neurona u tom sloju u svakom prolazu unapred. Ovo sprečava da se mreža
osloni na uzak skup "specijalizovanih" neurona (co-adaptation) i primorava je
da nauči redundantnu, robusniju reprezentaciju — efikasno deluje kao
implicitno usrednjavanje (ensembling) eksponencijalno mnogo pod-mreža.
Vrednost 0.5 je standardna, najčešće korišćena vrednost iz originalnog
Dropout rada (Srivastava et al., 2014) za FC slojeve.

**Zašto FC glava ima dva sloja (256→128→2) a ne direktno 256→2?**
Jedan dodatni skriveni sloj sa nelinearnošću (ReLU) dozvoljava mreži da
kombinuje 256 usrednjenih osobina na nelinearan način pre finalne odluke —
veća izražajna moć klasifikatora bez značajnog rasta broja parametara
(256×128 + 128×2 ≈ 33.000 parametara, zanemarljivo u odnosu na konvolucioni deo).

**Zašto se dropout stavlja i pre prvog i pre drugog FC sloja?** Regularizacija
se primenjuje na oba prelaza kroz FC deo mreže jer je upravo FC deo (gusto
povezan, veliki broj veza) najosetljiviji na overfitting; konvolucioni deo je
već regularizovan kroz weight sharing i BatchNorm.

### 3.5. Zašto 2 izlazna neurona (softmax/CrossEntropy) umesto 1 neuron + sigmoid?

Mreža vraća 2 logita (za klase "nije vozilo" i "vozilo"), a ne jedan logit sa
sigmoidom. Matematički su ova dva pristupa za binarnu klasifikaciju
ekvivalentna (softmax nad 2 klase svodi se na sigmoid razlike logita), ali
2-izlazni pristup uz `nn.CrossEntropyLoss` je standardni PyTorch idiom, čini
kod direktno proširivim na višeklasnu klasifikaciju (npr. ako bi se kasnije
htelo razlikovati *tip* vozila) i simetrično tretira obe klase.

### 3.6. Inicijalizacija težina — zašto Kaiming (He) inicijalizacija?

```python
nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # Conv2d
nn.init.ones_(bn.weight); nn.init.zeros_(bn.bias)                        # BatchNorm
nn.init.normal_(linear.weight, 0, 0.01)                                   # Linear
```

**Zašto Kaiming a ne Xavier/Glorot za konvolucione slojeve?** Xavier
inicijalizacija je izvedena pod pretpostavkom linearne (ili tanh)
aktivacije; ReLU "seče" pola raspodele na nulu, pa bi Xavier inicijalizacija
sistematski smanjivala varijansu aktivacija kroz duboke mreže. Kaiming
inicijalizacija je izvedena specifično za ReLU i skalira težine tako da
varijansa aktivacija ostane približno konstantna kroz slojeve, čime se
sprečava da signal (i gradijent) eksponencijalno "zgasne" ili "eksplodira" s
dubinom mreže. `mode="fan_out"` bira skaliranje na osnovu broja izlaznih
veza — pogodno kad je fokus na očuvanju varijanse gradijenta unazad kroz mrežu.

**Zašto BatchNorm počinje sa weight=1, bias=0?** Ovo znači da BatchNorm sloj
na početku treninga radi kao identična transformacija (samo normalizuje, ne
skalira niti pomera) — mreža počinje iz neutralne, predvidljive tačke i sama
uči (kroz trening) koliko treba da odstupi od toga.

**Zašto FC slojevi koriste malu normalnu raspodelu (std=0.01)?** Standardna,
konzervativna inicijalizacija za klasifikacioni sloj na kraju mreže — male
početne težine drže početne logite blizu nule (izlaz blizu uniformne
raspodele verovatnoća 50/50), što daje stabilan početak treninga bez
ekstremnih početnih gubitaka.

---

## 4. Funkcija gubitka i optimizacija

### 4.1. `nn.CrossEntropyLoss()`

Standardni izbor za višeklasnu (ovde: dvoklasnu) klasifikaciju. Interno
kombinuje `log_softmax` + negativni log-verovatnoća (NLL), što je numerički
stabilnije od ručnog računanja softmax-a pa zatim log-a odvojeno.

### 4.2. Optimizator — zašto `AdamW`?

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

**Zašto Adam (adaptivni) umesto običnog SGD?** Adam održava pokretne proseke
prvog i drugog momenta gradijenta za svaki parametar posebno, efektivno
prilagođavajući learning rate svakom parametru individualno. Ovo mreži
omogućava brzu, stabilnu konvergenciju bez pažljivog ručnog podešavanja
learning rate rasporeda, što je posebno korisno za projekat sa ograničenim
vremenom/resursima za eksperimentisanje sa hiperparametrima. (Čist SGD+momentum
može ponekad dati blago bolju finalnu generalizaciju uz mnogo pažljiviji LR
schedule, ali zahteva mnogo više podešavanja.)

**Zašto AdamW a ne "obični" Adam?** Kod originalnog Adama, "weight decay"
(L2 regularizacija) se implementira tako što se doda gradijentu pre
računanja adaptivnih momenata — što znači da se regularizacija *sama*
skalira adaptivnim faktorima učenja, na neintuitivan i teško-kontrolisan
način. AdamW **razdvaja (decouples)** weight decay od gradijentnog koraka:
decay se primenjuje direktno na težine, nezavisno od adaptivnog skaliranja
gradijenta. Ovo je pokazano (Loshchilov & Hutter, 2019 — "Decoupled Weight
Decay Regularization") da daje bolju i predvidljiviju regularizaciju i bolju
generalizaciju u odnosu na klasični Adam sa L2 regularizacijom.

**Zašto `weight_decay=1e-4`?** Standardna, umerena vrednost iz literature za
CNN-ove ove veličine — dovoljno jaka da blago penalizuje velike težine i
pomogne generalizaciji, ali dovoljno slaba da ne uguši kapacitet mreže od
1,18M parametara.

**Zašto `lr=1e-3`?** Standardna početna vrednost za Adam/AdamW (podrazumevana
vrednost u većini literature i frameworkova). Dovoljno velika za brzu
konvergenciju u ranim epohama, a scheduler (videti dalje) je postepeno
smanjuje kako trening odmiče.

### 4.3. `CosineAnnealingLR` scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
```

**Zašto scheduler uopšte?** Konstantan learning rate kroz ceo trening je
suboptimalan: veći LR je koristan na početku (brzo napredovanje kroz
loss-površ), ali ka kraju treninga ometa fino podešavanje (konvergenciju ka
minimumu) i može uzrokovati "poskakivanje" oko optimuma.

**Zašto baš kosinusni raspored (a ne, npr., step decay)?** Cosine annealing
glatko i kontinuirano smanjuje LR po kosinusnoj krivi od početne vrednosti do
skoro nule, bez naglih "skokova" karakterističnih za step-decay raspored
(gde se LR naglo prepolovi na određenim epohama). Glatko opadanje je
empirijski pokazalo bolju i stabilniju konvergenciju u mnogim CNN
eksperimentima (Loshchilov & Hutter, "SGDR", 2017), a i jednostavnije je za
podešavanje — nema dodatnih hiperparametara (kada i koliko smanjiti LR) osim
ukupnog broja epoha.

**Zašto `T_max=cfg.epochs` (dakle = 20)?** Ovim se kosinusna kriva "isteže"
tako da LR dostigne svoj minimum tačno na poslednjoj epohi treninga — ceo
budžet epoha se iskoristi za postepeno smanjenje LR, bez "traćenja" epoha na
već-nizak LR pre kraja ili naglog prekida pre nego što LR stigne do minimuma.

### 4.4. Zašto se `scheduler.step()` poziva posle svake epohe (ne posle svakog batch-a)?

Ovo je standardna praksa za epoch-based schedulere: LR treba da se menja na
grubljoj vremenskoj skali (epoha) da bi se prosečno stanje optimizacije kroz
celu epohu ocenilo pre sledeće promene, umesto da LR fluktuira u sred jedne
epohe.

---

## 5. Trening petlja i preostali hiperparametri

Vrednosti podrazumevane u `src/config.py` / `train.py` (mogu se promeniti
preko CLI argumenata):

| Hiperparametar | Vrednost | Zašto |
|---|---|---|
| `batch_size` | 128 | Kompromis: dovoljno veliki batch da BatchNorm statistike (srednja vrednost/varijansa po batch-u) budu stabilne i da se iskoristi paralelizam GPU-a, a dovoljno mali da stane u memoriju na tipičnom hardveru (CPU/MPS/consumer GPU) i da zadrži koristan "šum" u proceni gradijenta (koji sam po sebi blago regularizuje trening). |
| `epochs` | 20 | Dovoljno da kriva treninga i validacije konvergira i "zaravni" (videti `checkpoints/history.json` — posle epohe ~15 poboljšanja su marginalna) bez nepotrebnog trošenja vremena/resursa. |
| `val_split` | 0.1 | 10% trening skupa (5.000 slika) je dovoljno da pouzdano proceni koji epoch/checkpoint je najbolji, a da se ne "žrtvuje" previše podataka za sam trening. |
| `seed` | 42 | Reproducibilnost eksperimenta (ista podela podataka, ista inicijalizacija mreže pri ponovljenom pokretanju). |
| `num_workers` | 4 | Broj paralelnih CPU procesa za učitavanje/augmentaciju slika u pozadini dok GPU/CPU trenira — sprečava da učitavanje podataka bude usko grlo. Na CPU/MPS se preporučuje 0 (izbegavanje overhead-a multiprocesinga za mali dataset). |

**Zašto `drop_last=True` samo za trening loader (ne i za val/test)?**
Ako je poslednji batch u epohi mnogo manji od 128 (npr. samo par uzoraka),
statistike Batch Normalization-a računate na tom mini-batch-u bi bile vrlo
nepouzdane (velika varijansa procene) i unosile bi nestabilnost u trening.
Za validaciju/test ovo nije problem jer se `model.eval()` koristi — BatchNorm
tada koristi **akumulirane (running) statistike** iz treninga, a ne statistiku
tekućeg batch-a, pa nema razloga da se odbaci deo test/val podataka.

**Zašto `model.train()` / `model.eval()` prekidači (u `src/engine.py`)?**
Dropout i BatchNorm se ponašaju različito u treningu i inferenciji: u
`train()` modu Dropout nasumično gasi neurone i BatchNorm koristi
batch-statistiku; u `eval()` modu Dropout je isključen (koriste se svi
neuroni) i BatchNorm koristi fiksnu, naučenu (running) statistiku iz celog
treninga — jer inferencija mora biti determinstička i ne sme zavisiti od toga
koji su drugi uzorci slučajno u istom batch-u.

**Zašto `optimizer.zero_grad()` pre svakog `backward()`?** PyTorch po
podrazumevanom podešavanju *akumulira* gradijente pri uzastopnim pozivima
`.backward()`; bez ručnog resetovanja na nulu, gradijenti iz prethodnog
mini-batch-a bi se sabirali sa novim, dajući pogrešan (uvećan i pomešan)
signal za `optimizer.step()`.

**Zašto se najbolji model bira po `val_acc` (a ne po poslednjoj epohi ili
`val_loss`)?** Model iz poslednje epohe nije nužno najbolji — moguće je da
mreža kasnije blago overfituje (val metrika stagnira/opadne dok trening
metrika i dalje raste, videti epohe 5–13 u `history.json` gde val_acc
osciluje pre nego što se stabilizuje). Čuvanjem checkpointa sa najvišom
validacionom tačnošću, projekat efektivno implementira jednostavan oblik
*early stopping / model selection*, direktno optimizujući metriku od
interesa (accuracy) umesto posredne (loss).

**Zašto se `criterion`, `mean`, `std` čuvaju unutar checkpointa (`best_model.pt`)?**
Da bi `predict.py` mogao kasnije da učita model i primeni *tačno iste*
statistike normalizacije korišćene pri treningu — bez toga bi inferencija na
novim slikama koristila pogrešnu normalizaciju i davala nepouzdane rezultate.

---

## 6. Rezultati

Na test skupu (10.000 slika, koji CNN i baseline modeli nikad nisu videli u
treningu):

| Model | Tačnost | Preciznost | Odziv | F1 | ROC-AUC | # parametara |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistička regresija | 0,816 | 0,791 | 0,733 | 0,761 | 0,879 | — |
| MLP (bez konvolucija, 256-128) | 0,880 | 0,854 | 0,845 | 0,850 | 0,943 | — |
| **VehicleCNN (ova mreža)** | **0,973** | **0,969** | **0,963** | **0,966** | **0,997** | 1.180.354 |

**Kriva treninga (`checkpoints/history.json`):** tačnost na treningu raste
skoro monotono od 89,4% (epoha 1) do 98,4% (epoha 20); validaciona tačnost
prati sličnu putanju sa nešto više oscilacija (npr. pad na epohi 6, 93,1%)
ali se stabilizuje oko 97,2–97,4% u poslednjih ~5 epoha. Razlika između
finalne trening tačnosti (98,4%) i validacione/test tačnosti (97,3%) je mala
(~1 procentni poen), što pokazuje da **overfitting nije značajan problem** —
dropout, weight decay i augmentacija su efikasno regularizovali mrežu.

**Zašto se poredi baš sa logističkom regresijom i MLP-om (a ne, recimo, sa
random forest-om ili SVM-om)?** Cilj poređenja nije "koji je najbolji
klasifikator uopšte", već da se **izoluje doprinos konvolucione arhitekture**:
1. Logistička regresija = najjednostavniji linearni klasifikator na sirovim
   pikselima (donja granica/sanity-check).
2. MLP (potpuno povezana mreža, 256-128 skrivenih neurona) = nelinearni
   klasifikator **istog opšteg tipa** (neuronska mreža, gradijentni trening)
   ali **bez** konvolucija/deljenja težina/lokalnosti.
3. CNN = ista filozofija kao MLP + konvoluciona induktivna pristrasnost
   (inductive bias).

Skok sa MLP (88,0%) na CNN (97,3%), uz da su oba neuronske mreže trenirane
gradijentnim spustom, izoluje da je **poboljšanje od ~9 procentnih poena
posledica konvolucione arhitekture same po sebi** (lokalnost + deljenje
težina + hijerarhijsko učenje osobina), a ne razlike u optimizacionom
algoritmu ili tipu modela uopšte.

---

## 7. Zašto se prate Precision/Recall/F1/ROC-AUC pored tačnosti?

Zbog blage neuravnoteženosti klasa (40% vozila / 60% ne-vozila u test skupu),
sama tačnost može delimično sakriti slabosti modela (npr. model koji uvek
predviđa "ne-vozilo" bi imao 60% tačnosti bez ikakve stvarne diskriminativne
moći). Zato se prate:

- **Preciznost (precision)** — od svih slika koje je model označio kao
  "vozilo", koliki procenat zaista jeste vozilo (meri lažno pozitivne).
- **Odziv (recall)** — od svih slika koje zaista jesu vozilo, koliki procenat
  je model uspeo da pronađe (meri lažno negativne).
- **F1** — harmonijska sredina preciznosti i odziva; jedan broj koji
  balansira oba tipa greške, koristan kad su obe greške podjednako bitne.
- **ROC-AUC** — meri kvalitet rangiranja/verovatnoća nezavisno od izabranog
  praga odluke (0,5); vrednost 0,997 znači da model gotovo savršeno
  razdvaja klase po dodeljenoj verovatnoći, bez obzira na tačan prag.

Sve četiri metrike kod CNN-a su međusobno vrlo blizu (0,963–0,997) i znatno
iznad oba baseline-a na svakoj pojedinačnoj metrici — što potvrđuje da
poboljšanje nije artefakt jedne metrike (npr. da model "vara" tako što
favorizuje većinsku klasu), već je konzistentno robusno poboljšanje.

---

## 8. Inferencija na novim slikama (`predict.py`)

```python
transforms.Resize((32, 32))
transforms.ToTensor()
transforms.Normalize(mean, std)  # iz checkpointa
```

**Zašto `Resize((32,32))`?** Mreža je trenirana isključivo na 32×32 ulazima
(zbog `AdaptiveAvgPool2d`, tehnički bi mogla da primi i drugu rezoluciju, ali
naučene težine — posebno prvi konv sloj — su kalibrisane za statistiku 32×32
CIFAR-10 slika). Da bi inferencija na proizvoljnoj korisničkoj slici (npr.
JPG sa telefona) bila validna, ulaz se prvo svodi na istu rezoluciju kao u
treningu.

**Zašto se koriste `mean`/`std` sačuvani u samom checkpointu, a ne oni iz
`TrainConfig`-a direktno?** Ovo čini `predict.py` nezavisnim od trenutnog
stanja `config.py` — čak i ako se konfiguracija u budućnosti promeni, stari
checkpoint i dalje nosi tačne statistike sa kojima je treniran, garantujući
konzistentnost između treninga i inferencije.

---

## 9. Inženjerske/tehničke odluke (manje, ali mogu da pitaju)

- **`get_device()`** bira automatski `CUDA` → `MPS` (Apple Silicon) → `CPU`,
  po opadajućem redosledu očekivane brzine — kod je prenosiv između mašina
  bez ručne izmene.
- **`pin_memory=(device.type == "cuda")`** — "pinned" (stranicama-zaključana)
  memorija ubrzava transfer podataka sa CPU-a na GPU (samo relevantno za
  CUDA; MPS/CPU od toga nemaju korist pa se ne koristi).
- **`non_blocking=True`** pri `.to(device)` pozivima — omogućava
  asinhrono kopiranje podataka na GPU dok CPU priprema sledeći batch
  (preklapanje I/O i računanja), efekat samo uz `pin_memory=True`.
- **`torch.no_grad()`** dekorator na `evaluate`/`collect_predictions` —
  isključuje računanje i čuvanje gradijenata tokom evaluacije, čime se
  štedi memorija i ubrzava unapredni prolaz (nema potrebe za gradijentima
  kad se težine ne ažuriraju).
- **`StandardScaler`** u `baselines.py` (fit samo na trening skupu, zatim
  primenjen i na trening i na test) — logistička regresija i MLP su
  osetljivi na skalu ulaznih osobina (za razliku od CNN-a, koji kroz
  BatchNorm ima ugrađenu neku vrstu adaptivne normalizacije); fit
  isključivo na trening podacima sprečava "curenje" informacija (data
  leakage) iz test skupa u proces pripreme podataka.

---

## 10. Moguća pitanja profesora/asistenta i predlog odgovora

**P: Zašto CNN, a ne neka klasična metoda mašinskog učenja (npr. SVM, random
forest) na sirovim pikselima?**
O: Klasične metode ne poseduju induktivnu pristrasnost za prostornu strukturu
slika — tretiraju sliku kao ravan vektor brojeva, gubeći informaciju o
susedstvu piksela. CNN koristi lokalnu povezanost i deljenje težina da uči
hijerarhiju osobina (ivice → teksture → delovi oblika → objekat), što
projekat direktno demonstrira poređenjem sa logističkom regresijom (0,816)
i MLP-om (0,880) na istim sirovim pikselima — CNN dostiže 0,973.

**P: Zašto baš 3 konvoluciona bloka, a ne 2 ili 5?**
O: Sa ulazom 32×32, tri MaxPool(2) sloja smanjuju rezoluciju na 4×4
(32→16→8→4), gde 4×4 mapa osobina sa 256 kanala i dalje nosi dovoljno
prostorne informacije za Global Average Pooling da bude smisleno (a ne
degeneriše se na 1×1 prerano). Dodatni (četvrti) blok bi sveo mapu na 2×2 ili
manje, gubeći previše prostorne rezolucije za tako mali ulaz; manje blokova
(2) bi ostavilo mrežu sa manjim receptivnim poljem i manjim kapacitetom za
apstrakciju. Tri bloka su uobičajen, dobro balansiran izbor za 32×32 slike u
literaturi (CIFAR-stil CNN-ovi).

**P: Kako sprečavate overfitting?**
O: Kombinacijom više mehanizama koji deluju na različitim nivoima:
augmentacija podataka (RandomCrop + Flip) uvećava efektivnu raznovrsnost
trening skupa; Dropout(0.5) u klasifikatoru sprečava kolinearnu
zavisnost neurona; `weight_decay=1e-4` (L2 regularizacija) penalizuje velike
težine; Batch Normalization takođe ima blag regularizacioni efekat; i konačno,
čuva se checkpoint sa najboljom validacionom tačnošću (efektivno rani
prekid), umesto poslednje epohe. Rezultat: razlika trening/validacija/test
tačnosti je svega ~1 procentni poen (98,4% / 97,3% / 97,3%).

**P: Zašto AdamW a ne SGD sa momentumom?**
O: AdamW adaptivno prilagođava learning rate svakom parametru pojedinačno na
osnovu istorije gradijenata, što ubrzava i stabilizuje konvergenciju uz
minimalno ručno podešavanje. AdamW dodatno ispravlja poznat problem klasičnog
Adama gde weight decay nije ispravno razdvojen od adaptivnog skaliranja
gradijenta, dajući bolju regularizaciju. SGD+momentum bi mogao dati
uporedivu ili blago bolju generalizaciju uz mnogo pažljiviji, ručno podešen
raspored learning rate-a — što nije bio prioritet za obim ovog projekta.

**P: Šta je Batch Normalization i zašto pomaže?**
O: Za svaki mini-batch, normalizuje aktivacije svakog kanala na srednju
vrednost 0 i varijansu 1 (a zatim uči sopstveno skaliranje/pomeranje —
`weight`/`bias`), čime se sprečava da raspodela ulaza u svaki sloj drastično
"luta" tokom treninga (internal covariate shift). Ovo dozvoljava veći,
stabilniji learning rate i bržu konvergenciju.

**P: Objasnite razliku između `model.train()` i `model.eval()`.**
O: Utiče na ponašanje Dropout-a (aktivan/nasumičan u train, isključen u eval)
i BatchNorm-a (koristi statistiku tekućeg batch-a u train, koristi fiksnu
naučenu running-statistiku u eval) — nužno za deterministički, ponovljiv
rezultat pri inferenciji.

**P: Zašto je test tačnost (0,973) veoma blizu validacionoj (najbolja 0,974)?**
O: Jer je model biran (checkpoint) upravo po validacionoj tačnosti, i
validacioni skup je dovoljno velik (5.000 slika) i reprezentativan da dobro
proceni performansu na potpuno nezavisnom test skupu — što je znak da model
generalizuje, a ne da je "nameštanje" (overfitting na validaciju).

**P: Da li je skup podataka balansiran? Da li to utiče na evaluaciju?**
O: Blago neuravnotežen — 40% vozila / 60% ne-vozila (posledica prirodne
podele: 4 od 10 originalnih klasa su vozila). Zbog toga se pored tačnosti
prate i preciznost, odziv, F1 i ROC-AUC, koje daju potpuniju sliku od same
tačnosti kada klase nisu 50/50.

**P: Zašto se koristi CrossEntropyLoss sa 2 izlazna neurona umesto
BCEWithLogitsLoss sa 1 izlaznim neuronom?**
O: Matematički ekvivalentno za binarnu klasifikaciju (softmax nad 2 logita
= sigmoid razlike logita), ali 2-neuronski/softmax pristup je standardniji
PyTorch idiom i direktno se uopštava na višeklasnu klasifikaciju bez izmene
arhitekture izlaznog sloja.

**P: Koliko parametara ima mreža i da li je to mnogo?**
O: 1.180.354 (≈1,18 miliona). Za poređenje, originalni VGG16 ima ~138
miliona parametara — ova mreža je namerno mnogo kompaktnija jer su ulazne
slike samo 32×32 (naspram 224×224 za ImageNet), pa nije potreban toliki
kapacitet; kompaktnost takođe smanjuje rizik od overfitting-a na relativno
mali dataset i ubrzava trening.

**P: Zašto se ne koristi transfer learning (npr. predtrenirani ResNet)?**
O: Cilj projekta je da se arhitektura i trening isprojektuju i implementiraju
samostalno (originalan kod), radi demonstracije razumevanja principa CNN-a
"od nule" — a ne da se osloni na tuđe predtrenirane težine. Ovo je i eksplicitno
navedeno u README-u ("sav izvorni kod je originalan").

**P: Kako biste unapredili model da imate više vremena/resursa?**
O: Nekoliko pravaca: (1) jača augmentacija (Cutout/MixUp/CutMix) za dodatnu
regularizaciju; (2) dublja/šira mreža ili rezidualne veze (ResNet-stil) ako
bi dataset bio veći; (3) sistematska pretraga hiperparametara (learning rate,
weight decay, dropout stopa) umesto ručno odabranih standardnih vrednosti;
(4) k-struka unakrsna validacija (cross-validation) za pouzdaniju procenu
varijanse rezultata; (5) kalibracija verovatnoća (npr. temperature scaling)
ako bi izlazne verovatnoće trebalo da budu dobro kalibrisane, a ne samo
tačan poredak (za šta ROC-AUC već ukazuje da model dobro radi).

**P: Šta predstavlja receptivno polje i koliko iznosi na izlazu iz
konvolucionog dela?**
O: Receptivno polje je oblast ulazne slike koja utiče na jedan izlazni
neuron. Svaki 3×3 konv sloj uvećava receptivno polje za 2 piksela; sa dva
konv sloja po bloku i tri MaxPool(2) sloja (koji dodatno duplaju efektivni
"korak" receptivnog polja svakog sledećeg sloja), receptivno polje na kraju
konvolucionog dela pokriva celu ulaznu sliku (32×32) — poslednji sloj
"vidi" celu sliku, što opravdava da je globalni average pooling smislen
(svaki od 256 kanala već je agregirao informaciju sa cele slike).

---

## 11. Kratak "elevator pitch" (za uvod u odbranu)

> "Napravio sam konvolucionu neuronsku mrežu u VGG stilu za binarnu detekciju
> vozila na CIFAR-10 skupu, gde su avion/automobil/brod/kamion označeni kao
> 'vozilo' a ostalih 6 klasa kao 'nije vozilo'. Mreža ima 3 konvoluciona
> bloka (svaki sa dva 3×3 konv sloja, batch normalizacijom i max poolingom),
> global average pooling i mali potpuno povezani klasifikator sa dropout
> regularizacijom — ukupno 1,18 miliona parametara. Trenirana je AdamW
> optimizatorom sa kosinusnim opadanjem learning rate-a, uz augmentaciju
> podataka (random crop i horizontalni flip) da spreči overfitting. Na test
> skupu od 10.000 slika postiže 97,3% tačnosti i 0,997 ROC-AUC, što je
> značajno bolje od logističke regresije (81,6%) i potpuno povezane mreže
> bez konvolucija (88,0%) — dokazujući da je konvoluciona arhitektura, a ne
> samo nelinearnost, ključni faktor uspeha."

---

*Napomena: svi brojevi u ovom dokumentu (metrike, broj parametara, istorija
treninga) su preuzeti direktno iz `reports/comparison.json`,
`reports/evaluation.json` i `checkpoints/history.json` generisanih u ovom
projektu — nisu izmišljeni ni aproksimirani.*
