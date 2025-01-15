# Grasping via (almost) plain PPO

<center>
<img src="./video/show.gif"/>
</center>

## Commands

Installation (I hope that didn't forget any requirements, it won't work without cuda):
```bash
pip install -r requirements.txt
```

#### Train
(it uses ```conf.yaml``` by default)
**It will delete all previous checkpoints, if run with load=False**
```bash
python3 main.py
```
*Tensorboard logs are located in outputs/{date}/{time}/*


#### Show demo
(requires to have any checkpoint in "checkpoint/" folder it will be created automatically
with train, by default it will use the checkpoint with the greatest amount of "steps"):
```bash
python3 main.py --config-name show
```
#### Record video
(like above, also requires any checkpoint)
```bash
python3 main.py --config-name record
```
