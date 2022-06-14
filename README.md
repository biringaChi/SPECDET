SPECDET: Fast Spectre Vulnerabilities and Attack Detection via Gadgets and CPU-Processes State
---
Installation
---
```
$ pip install -r requirements.txt
$ git clone https://github.com/biringaChi/SPECDET
```
Gadget Data
---
Refer: [Gadgets](https://github.com/biringaChi/SPECDET/tree/main/datasets/spectre_gadgets)

CPU-PS Data
---
Refer: [CPU-PS](https://github.com/biringaChi/SPECDET/tree/main/datasets/cpu_processes)

Generating Spectre Embeddings
---
```
$ cd src/SGDetector/
$ python spectre_embed.py
```
Train & Test Vulnerability Detector
---
```
$ cd src/SGDetector/
$ python train.py --epochs=<arg> --lr=<arg> --batch_size=<arg>
```
Evaluate
---
```
$ cd src/SGDetector/
$ python test.py
```

Train & Test Attack Detector
---
```
$ cd src/CPSDetector/
$ python train.py
```
<!--Problem using SPECDET?-->
<!------->
<!--Open an issue.-->

To cite
---
```
@inproceedings{coming soon...}
```