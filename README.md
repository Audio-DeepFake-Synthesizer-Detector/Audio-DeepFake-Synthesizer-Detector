# Audio DeepFake Synthesizer Detector
In questa repository è presente un modello CNN in formato h5, sviluppato in python, in grado di distinguere gli audio genuini da quelli fake.
Questo modello è stato realizzato e portato come progetto per l'esame di Cybersecurity M dell'unversità di Bologna.

## Come funziona?
Il modello è stato realizzato con il framework Tensorflow.keras e addestrato con due Dataset dai quali sono state estratte le features, necessarie per l'addestramento, utilizzando la libreria librosa di python.
Il modello riesce ad identificare con una buona percentuale di accuratezza gli audio passati.
![Senzanome](https://user-images.githubusercontent.com/100919731/213255929-d9dac9f0-49de-40f2-8055-d7bca2897801.png)

## Come utilizzarlo
L'utilizzo del modello è estremamente intuitivo (utilizzare il file predict.py come esempio).
Come integrare il modello nel proprio progetto:
### Fase 1:
Estrarre dal file txt fornito (Extracted_Features_Final.txt) le features audio

![image](https://user-images.githubusercontent.com/100919731/213257836-cdb969a3-8bbe-432f-90bc-3c8e00699177.png)
### Fase 2:
Caricare ed utilizzare il modello

![image](https://user-images.githubusercontent.com/100919731/213258464-c6423a04-9372-4279-9543-182736b22f5f.png)

## Dataset
Per quanto riguarda la base di dati ci siamo affidati ai dataset di [ASVspoof](https://www.asvspoof.org/index2021.html) per gli audio fake  e a [Common Voice](https://commonvoice.mozilla.org/it/datasets) per gli audio genuini

##Credits
-[Giorgio Mocci](https://github.com/giorgio-mocci)
-[Marco Motamed](https://github.com/MotaMarco)
