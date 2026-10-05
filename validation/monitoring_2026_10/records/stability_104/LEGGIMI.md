# 104 — verifica cieca della stabilità degli esempi

Regola congelata: 2026-10-04T19:59:35.497571+00:00. Nessuna esecuzione Tinker effettuata nella preparazione.

Unica domanda: l'ordine di stabilità degli stessi 32 esempi, stimato su 103 e 106, si trasferisce al 104?

- Finestra: stati 300–399, cento osservazioni e 99 differenze consecutive.
- Stabilità: mediana del valore assoluto delle differenze di loss di ogni esempio.
- Riferimento: media dei ranghi calcolati separatamente nei due run di calibrazione.
- Criterio principale: correlazione di Spearman con il 104 almeno 0,50. È una soglia pratica scelta prima del test; non equivale a significatività statistica.
- Candidati selezionati, numerazione da 1 a 32: 22, 10, 13, 7, 9, 19, 23, 30. L'eventuale coincidenza con gli otto più stabili del 104 è solo descrittiva.
- La correlazione 103–106 è 0.767229; questo è un dato di calibrazione, non un risultato del 104.
- La semplice loss bassa è una spiegazione alternativa: viene riportato anche un confronto descrittivo con il suo ordinamento, senza cambiare il criterio principale.

Restano invariati modello Qwen3-8B, rango fisso 8, learning rate 1e-4, 400 aggiornamenti, batch 8, 32 esempi di controllo, funzioni di training e calendario originale di raccolta. Il controller adattivo è assente, come nei due run sani. Non è un test dell'efficacia di Unified-LoRA.

Il codice originale healthy_core.py e gli esempi di controllo sono conservati byte per byte. Il nuovo collector modifica il manifesto, autorizza soltanto il 104 dopo il congelamento ed esclude l'analisi dei recuperi. Le funzioni che effettuano training, forward, generazione e controlli di integrità sono identiche all'originale.

La raccolta comprende fino a 4.352 risposte generate per mantenere il protocollo dei run precedenti. Il test primario usa solo le loss per esempio. La generazione viene archiviata e non entra nel verdetto. Il 104 viene utilizzato per questa domanda; eventuali future regole costruite dopo averne visto i risultati richiederanno un altro holdout.

Esiti: CRITERIO_DI_REPLICA_RAGGIUNTO, CRITERIO_DI_REPLICA_NON_RAGGIUNTO oppure NON_VALUTABILE. Arresti e dati incompleti non si trasformano in risultati negativi e non attivano automaticamente nuove esecuzioni.

Limiti: un run sano non stima falsi allarmi, non prova sensibilità al degrado, non dimostra previsione del recupero o benefici sul fluido. La stabilità tipica non limita i picchi rari.

Hash della regola: 5806b5e72401b8a6aecdbc9d284a9f6581cffa34faafbc5568e6762a07eddf78.
Hash del protocollo di raccolta: 41c4e612a1a7765a697ca93355dfeaee8249e22f1272cde97c7873f7b15da498.
