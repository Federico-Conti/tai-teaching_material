## Confronto tra Attacchi Maximum-Confidence e Minimum-Norm

Immaginiamo di aver calcolato questi due attacchi su un modello **A**, e poi di testare gli stessi esempi su un modello **B** (addestrato sugli stessi dati ma con piccole differenze nel boundary).

- Gli attacchi **PDG maximum-confidence**, viaggiando più in profondità nella regione sbagliata, hanno maggiore probabilità di trasferirsi (**transferability**): restano efficaci anche su un modello diverso ma addestrato sugli stessi dati.
  - Migliore per testare la robustezza di altri modelli o modelli “ignoti” (**black-box**), perché non si adatta al confine ma “ignora” la geometria locale e va in profondità.
  
- Gli attacchi **FMN minimum-norm** si adattano alla forma del confine su cui sono stati generati. Se il confine cambia anche solo un po’, l’attacco potrebbe non funzionare più.
  - non si omposta epsilon 
  -  Ottimo per analizzare la robustezza del modello che possedete (fornisce per ogni campione la distanza esatta dal confine).

### Trinità degli Attacchi Evasion

tutte queste implementazioni condividono la stessa struttura di base: 

(1) definire una loss (da minimizzare o massimizzare), 
(2) calcolare il gradiente di quella loss rispetto all’input, 
(3) aggiornare l’input usando un passo (gradient step). 

Poi opzionalmente si proietta o si clampa. 

### Manipolazione dell'Input

**Untargeted**

Voglio che il modello sbagli la previsione, qualsiasi classe va bene tranne quella vera.

**Targeted**

Voglio che il modello predica esattamente una classe specifica (la target class).

Il **gradiente** rappresenta la direzione in cui la loss aumenta. Per manipolare l’input e ottenere una previsione desiderata, è necessario scegliere in che direzione muoversi.

---

### Caso 1: Target (PDF e Minimum Norm Targeted)

```python
# Opzione 1
loss = - CrossEntropyLoss(outputs, y_target)
delta = delta + alpha * grad
```

- Il `-` davanti alla loss trasforma l'obiettivo: invece di aumentare la loss rispetto a `y_target`, la vuoi minimizzare, avvicinando la previsione a `y_target`.
- Il gradiente guida verso `y_target` e il passo è positivo.

```python
# Opzione 2
loss = CrossEntropyLoss(outputs, y_target)
delta = delta - alpha * grad
```

- Loss regolare, quindi il gradiente punta lontano da `y_target`.
- Invertendo il passo (`- alpha * grad`), ci si muove verso `y_target`.

---

### Caso 2: Untargeted (PGD e Minimum Norm Untargeted)

```python
# Opzione 1
loss = CrossEntropyLoss(outputs, y)
delta = delta + alpha * grad
```

- Vuoi massimizzare la loss rispetto alla classe vera, quindi sali nella direzione del gradiente.
- Ti allontani da `y`, come desiderato.

```python
# Opzione 2
loss = - CrossEntropyLoss(outputs, y)
delta = delta - alpha * grad
```

- Invertendo la loss, si ottiene lo stesso effetto: ci si muove lontano dalla classe corretta.
- Il gradiente punta verso `y`, ma il passo invertito porta lontano da `y`.

## Valutazioni di Robustezza

### Attacco PGD (Projected Gradient Descent)

complessità dell’ordine di O(N³) per ogni epsilon. 

- dove N è il numero di pesi. 

Per ogni campione nel test set:

- Calcola un forward pass → ottiene la predizione.
- Calcola un backward pass → ottiene il gradiente della loss rispetto all'input, per sapere in quale direzione perturbare l'immagine.
- Aggiorna il campione in base al gradiente.
- Ripete tutto per il numero di iterazioni.


Per ogni ε, attacchi tutto il test set.
Non sai esattamente quando ogni singolo campione cede

Ottieni una curva dove sull'asse x hai ε, sull’asse y hai la percentuale di campioni che resistono a quell’ε.

Esempio:
A ε = 0.6, il 20% dei campioni è robusto.

- Ma non so quali campioni sono robusti o quanto poco bastava per evaderli.

### Attacco FMN (Fast Minimum Norm)

Poiché non imposto mai un vincolo su epsilon, mi basta calcolarlo una volta, perché l’attacco stesso stima epsilon.

Richiede gestire i fallimenti degli attacchi.

Come si costruisce la curva di security evaluation con FMN?
Invece di avere una lista di accuracies per vari ε (come con PGD)

Per ogni valore di ε, la robust accuracy è la percentuale di campioni che resistono a perturbazioni più piccole di ε.

```sh
robust_accuracy = [(distanze < d).sum() / totale for d in distanze_ordinate]
```

- Ordini tutti i campioni in base alla loro ‖δ‖: la distanza minima necessaria per ingannarlo
- Per ogni distanza d, calcoli: quanti campioni sono robusti a perturbazioni minori di d?
- Questo ti dà la robust accuracy al variare del budget senza dover rieseguire l’attacco.
- Se FMN non riesce a ingannare il modello, allora quel campione è probabilmente robusto.
  - In quel caso si assegna ‖δ‖ = ∞ (o un valore molto alto)


la curva non è una media su tanti punti, ma mostra ogni singola perdita di robustezza campione per campione.

Esempio:

A ε = 0.6, il 20% dei campioni richiede una perturbazione più grande per essere ingannato

- So esattamente per ogni campione quanto devi perturbare per evaderlo.

