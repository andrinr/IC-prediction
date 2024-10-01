import jax
import jax.numpy as jnp
import json

class Metric:
    train_mse : list
    val_mse : list
    val_RSE : list
    val_RAE : list
    time : list

    def __init__(self):
        self.train_mse = []
        self.val_mse = []
        self.val_RSE = []
        self.val_RAE = []
        self.time = []

    def update(
            self,
            train_mse : float,
            val_mse : float,
            val_RSE : float,
            val_RAE : float,
            time : float):
        
        print(time)

        self.train_mse.append(train_mse)
        self.val_mse.append(val_mse)
        self.val_RSE.append(val_RSE)
        self.val_RAE.append(val_RAE)
        self.time.append(time)

    def to_dict(self):
        return {
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "val_RSE": self.val_RSE,
            "val_RAE" : self.val_RAE,
            "time": self.time
        }

    def toJSON(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=4)