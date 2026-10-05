# Protocollo 107: verifica prospettica della meccanica

Congelato: 2026-10-05T08:20:21.885375+00:00.
Questo documento completa il registro preliminare del seme 107. Nessuna esecuzione reale è avvenuta nella preparazione.

## Portata e criterio di completamento

Si verifica che le regole producano una decisione al momento del calo e che la decisione resti immutata dopo aver visto il futuro. Il 107 è un solo run sano: non assegna un vincitore, non dimostra superiorità predittiva e non stima una specificità di popolazione.

La verifica meccanica è completata soltanto con 450 aggiornamenti validi, raccolta completa, tokenizzazione confrontabile con la calibrazione, almeno un episodio e riproduzione esatta delle decisioni registrate online. Il criterio tecnico non richiede previsioni corrette: anche previsioni sbagliate possono essere eseguite correttamente. Gli errori vengono riportati separatamente.

Nessun episodio a D=2 => NON_VALUTABILE per la meccanica sugli episodi; lo stato della raccolta viene comunque riportato. Dati incompleti o problemi di integrità/comparabilità => NON_VALUTABILE. Una sola categoria osservata non permette il confronto fra categorie. Anche se comparissero entrambe, il confronto di superiorità resta non valutabile nel singolo 107.

## Training e raccolta

- Solo seme 107, 450 aggiornamenti, Qwen/Qwen3-8B, rango LoRA fisso 8, LR 0,0001, batch 8, SDK Tinker 0.29.0. Nessun controller adattivo o stress.
- Stessi 32 esempi di controllo. I 3.600 esempi di training sono esclusi dal controllo; i primi 3.200 coincidono con quelli che produrrebbe il vecchio generatore con seme 107 e 400 aggiornamenti.
- Loss media e 32 loss individuali a ogni stato 0–450.
- Generazione ogni 10 aggiornamenti: tre passaggi sullo stesso export e mediana delle risposte intere corrette su 32. Il denominatore resta 32.
- Probe da un secondo export agli stati 0,100,200,300,400,450; checkpoint completi a 100,200,300,400,450.
- Trigger diagnostico originale: dal 100 al 400, loss maggiore di mediana dei 20 stati precedenti più max(6 MAD non scalato,0,05); pausa 50, massimo quattro finestre, extra al trigger e cinque passi dopo. Gli extra non entrano in episodi, decisioni o etichette.
- Budget massimo: 4.416 risposte ordinarie, 192 risposte dei secondi export, 256 extra: **4.864 risposte**. Il notebook non lancia altri semi.
- Errori tecnici, interruzioni e loss non finite conservano dati parziali; non attivano riavvii. Un arresto numerico non diventa un mancato recupero.

## Episodio e momento della decisione

Soglia primaria **D=2 risposte**. È la soglia già descritta nella calibrazione che offre sei episodi completi per verificare la meccanica. Non è stata selezionata per massimizzare l'accuratezza delle previsioni.

Il riferimento è la mediana dei punteggi ai tre stati regolari precedenti (t−30,t−20,t−10); deve essere almeno 8. Un episodio inizia al primo stato regolare ammissibile in cui riferimento meno punteggio corrente è **maggiore o uguale a 2**. Non si ammette un nuovo episodio fino a t+50 incluso. L'ultimo inizio ammissibile è 400; 401–450 sono solo seguito per gli episodi primari.

Tutte le regole decidono dopo la valutazione generativa allo stato t e **prima dell'aggiornamento t**, che porterà allo stato t+1. Il log conserva stato, predizioni, caratteristiche, impronte dei prefissi di dati e collegamento al record precedente. La verifica offline ricalcola la stessa decisione utilizzando solo il prefisso fino a t. Il futuro non può rivederla.

## Cinque regole operative, stessa finestra

Le regole sono **euristiche esplorative nuove per questo test**, non componenti validate del controller Unified-LoRA. La calibrazione non contiene mancati recuperi completi: non si addestra un classificatore a due categorie.

Per ogni decisione si usano gli stessi 20 stati t−19…t: 15 passati (t−19…t−5) e 5 recenti (t−4…t). Per ogni esempio si calcola max(0, mediana recente − mediana passata).

1. **Esempi stabili:** mediana di questi incrementi sugli otto selezionati nella calibrazione. Numerazione 1–32: 22, 10, 13, 7, 9, 19, 23, 30.
2. **Loss media:** max(0, mediana recente − mediana passata) della loss media pesata per token già registrata.
3. **Variabilità della loss media:** max(0, deviazione standard di popolazione recente − deviazione standard di popolazione passata), sulla stessa finestra 15+5.
4. **Esempi a loss bassa:** stessa formula della regola 1, selezione alternativa congelata. Numerazione 1–32: 22, 10, 13, 7, 19, 23, 30, 31.
5. **Recupererà sempre:** previsione costante del recupero, agli stessi episodi; controllo indispensabile quando i recuperi sono frequenti.

Le selezioni usano la media dei ranghi dei run 103 e 106 nella finestra 300–399, con indice crescente per sciogliere parità. Per la stabilità si ordina la mediana delle variazioni assolute consecutive; per il livello si ordina la loss mediana. I due gruppi condividono sette esempi: il confronto può quindi risultare poco separato.

Per ciascuna delle prime quattro regole, la soglia è il **massimo del suo punteggio sui sei episodi completi recuperati di 103/106**. Superarla strettamente produce la previsione NO_CONFIRMED_RECOVERY_WITHIN_50; altrimenti RECOVERED_WITHIN_50. L'ipotesi operativa è che un'anomalia più grande di tutte quelle osservate nei sei recuperi segnali un mancato recupero: è un'ipotesi non validata, non un fatto. Non controlla un tasso di falsi allarmi.

Soglie numeriche congelate:
- `low_loss_examples`: `0.029927404454519239`
- `mean_loss`: `0.02784513046848644`
- `mean_loss_variability`: `0`
- `stable_examples`: `0.0091052469844059188`

La soglia della variabilità è zero perché tutti e sei i punteggi di calibrazione, con questa formula, sono zero. Non la si corregge dopo aver visto il 107. L'assenza di errori sui sei casi usati per fissare i massimi sarebbe una conseguenza della taratura, non una validazione.

## Etichette e resoconto

L'etichetta è **recuperato entro 50 aggiornamenti**. Il riferimento dell'episodio resta fisso. Occorrono due controlli generativi futuri consecutivi almeno al riferimento meno una risposta, entrambi entro t+50. La conferma è il secondo controllo. In assenza di conferma, con tutta la finestra disponibile, l'etichetta è NO_CONFIRMED_RECOVERY_WITHIN_50. Non significa danno permanente.

Una finestra incompleta è CENSORED, anche se una conferma appare nel tratto disponibile, coerentemente con la calibrazione precedente. Arrivare a 450 rimuove la censura amministrativa degli episodi ammessi, non quella da errori o misure mancanti.

Si riportano tutti gli episodi e, per ciascuna regola, i conteggi di recuperi previsti correttamente, falsi mancati recuperi, mancati recuperi previsti correttamente, mancati recuperi non segnalati e censurati esclusi. Nessuna classifica automatica o soglia modificata sul 107. Episodi dello stesso run non sono repliche indipendenti.

## Pilota sotto stress separato, non eseguito dal notebook

Si registra una griglia di sviluppo sui semi **108 e 109**, ciascuno con tre intensità: LR **3×, 10×, 30×** il valore sano, ossia 0,0003, 0,001 e 0,003. Sono sei traiettorie fresche da 450 aggiornamenti. Il LR elevato si applica ai 20 aggiornamenti indicizzati 217–236; primo stato esposto 218; prima dell'aggiornamento 237 si ripristina 0,0001. Non si biforca da checkpoint, per evitare di confondere il ripristino con lo stress.

Questa è una griglia candidata di pilota, non una gamma già capace di produrre entrambe le categorie. Nessun braccio riceve un'etichetta a priori. Le divergenze numeriche restano separate. Se manca una categoria osservabile, il pilota è insufficiente: nessun risultato discriminativo e nessuna ricerca automatica di intensità.

Per la futura verifica si riserva la famiglia di semi **200–209**, distinta dal pilota. Il numero necessario di run, l'effetto minimo e il criterio comparativo devono essere fissati dopo il pilota e prima di aprire quei risultati. La regola potrà essere sviluppata sul pilota con versione e scelte dichiarate; il 107 non varrà come test cieco di una regola modificata successivamente. Il notebook corrente implementa solo il sano 107.

## Provenienza

Calibrazione numerica: 103 e 106. Il 105 è solo contesto esplorativo; il 104 non entra nel fitting. Le funzioni originali di tokenizzazione, loss, training, generazione e salvataggio sono riutilizzate; le modifiche riguardano durata, calendario esteso e registrazione delle decisioni. Questa è una nuova domanda di monitoraggio, senza rivendicazioni sull'efficacia del controller o sui fluidi.

- Bundle di calibrazione: `58de2e3c7dca7d5d7aedfdcca0fc47ccb7a1b523495ff5228be0be48f1a7f4c2`.
- Regola: `c40b2492597adc9dad12062051a8a476dd2d8c8ad2207353d2d362219bd21cfc`.
- Protocollo eseguibile: `5dce902d483572d8486940f8ada80d3906958f4e380af5dca3f385fc4000dc25`.
