## Confronto tra Attacchi Maximum-Confidence e Minimum-Norm

PGD (Maximum-Confidence): Cerca di massimizzare la perdita il più possibile; viaggia “in profondità” nella regione errata, allontanandosi dalla frontiera.

FMN (Minimum-Norm): Si ferma appena raggiunge il confine più vicino, adattandosi perfettamente alla forma locale della decision boundary.

Immaginiamo di aver calcolato questi due attacchi su un modello **A**, e poi di testare gli stessi esempi su un modello **B** (addestrato sugli stessi dati ma con piccole differenze nel boundary).

- Gli attacchi **minimum-norm** si adattano alla forma del confine su cui sono stati generati. Se il confine cambia anche solo un po’, l’attacco potrebbe non funzionare più.
- Gli attacchi **maximum-confidence**, invece, viaggiando più in profondità nella regione sbagliata, hanno maggiore probabilità di trasferirsi (**transferability**): restano efficaci anche su un modello diverso ma addestrato sugli stessi dati.

- **Minimum-Norm**: Ottimo per analizzare la robustezza del modello che possedete (fornisce per ogni campione la distanza esatta dal confine).
- **Maximum-Confidence**: Migliore per testare la robustezza di altri modelli o modelli “ignoti” (**black-box**), perché non si adatta al confine ma “ignora” la geometria locale e va in profondità.

- I **maximum-confidence attacks** sono come corridori spericolati: “andiamo veloce, ovunque!”.
- I **minimum-norm** sono intenditori raffinati di vini e musica: precisi, eleganti, ma efficaci solo dove conoscono il terreno.

## Trinità degli Attacchi Evasion

tutte queste implementazioni condividono la stessa struttura di base: 
(1) definire una loss (da minimizzare o massimizzare), 
(2) calcolare il gradiente di quella loss rispetto all’input, 
(3) aggiornare l’input usando un passo (gradient step). 

Poi opzionalmente si proietta o si clampa. 

## Manipolazione dell'Input

**Untargeted**

Voglio che il modello sbagli la previsione, qualsiasi classe va bene tranne quella vera.

**Targeted**

Voglio che il modello predica esattamente una classe specifica (la target class).

Il **gradiente** rappresenta la direzione in cui la loss aumenta. Per manipolare l’input e ottenere una previsione desiderata, è necessario scegliere in che direzione muoversi.

---

### Caso 1: Target (PDF e Minimum Norm Targeted)

```python
# Opzione 1
loss = -CrossEntropyLoss(outputs, y_target)
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
loss = -CrossEntropyLoss(outputs, y)
delta = delta - alpha * grad
```

- Invertendo la loss, si ottiene lo stesso effetto: ci si muove lontano dalla classe corretta.
- Il gradiente punta verso `y`, ma il passo invertito porta lontano da `y`.