# Audio DeepFake Synthesizer Detector English version
This repository contains a CNN model in h5 format, developed in python, that is able to distinguish between genuine and fake audio.
This model was created and brought as a project for the Cybersecurity M exam at the University of Bologna.

## How does it work?
The model was created with the Tensorflow.keras framework and trained with two datasets from which the necessary features were extracted, using the python library librosa.
The model is able to identify with a good accuracy rate the audio passed.
![Senzanome](https://user-images.githubusercontent.com/100919731/213255929-d9dac9f0-49de-40f2-8055-d7bca2897801.png)

## How to use it
The use of the model is extremely intuitive (use the predict.py file as an example).   
How to integrate the model into your project:
### Phase 1:
Extract the audio features from the provided txt file (Extracted_Features_Final.txt)

![image](https://user-images.githubusercontent.com/100919731/213257836-cdb969a3-8bbe-432f-90bc-3c8e00699177.png)
### Phase 2:
Load and use the model

![image](https://user-images.githubusercontent.com/100919731/213258464-c6423a04-9372-4279-9543-182736b22f5f.png)

## Dataset
Regarding the database, we relied on the [ASVspoof](https://www.asvspoof.org/index2021.html) datasets for fake audio and the [Common Voice](https://commonvoice.mozilla.org/it/datasets) datasets for genuine audio.

# Audio DeepFake Synthesizer Detector Versione italiana
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

## Credits
- [Giorgio Mocci](https://github.com/giorgio-mocci)
- [Marco Motamed](https://github.com/MotaMarco)
