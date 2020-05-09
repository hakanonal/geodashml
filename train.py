from environment import environment
import wandb

def train():
    e = environment()
    try:
        e.start()
    finally:
        e.end()


wandb.agent('hakanonal/geodashml/g2ftdi6z',function=train)    