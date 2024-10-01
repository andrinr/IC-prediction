import jax
import jax.numpy as jnp
import json

class Metric:
    train_mse : jax.Array
    val_mse : jax.Array
    normalized_val_mse : jax.Array
    cross_correlation : jax.Array
    time : jax.Array

    def __init__(self, n : int):
        self.train_mse = jnp.zeros(n)
        self.val_mse = jnp.zeros(n)
        self.normalized_val_mse = jnp.zeros(n)
        self.cross_correlation = jnp.zeros(n)
        self.time = jnp.zeros(n)

    def update(
            self,
            i : int,
            train_mse : float,
            val_mse : float,
            normalized_val_mse : float,
            cross_correlation : float,
            time : float):

        self.train_mse = self.train_mse.at[i].set(train_mse)
        self.val_mse = self.val_mse.at[i].set(val_mse)
        self.normalized_val_mse = self.normalized_val_mse.at[i].set(normalized_val_mse)
        self.cross_correlation = self.cross_correlation.at[i].set(cross_correlation)
        self.time = self.time.at[i].set(time)

    def to_dict(self):
        return {
            "train_mse": self.train_mse.tolist(),
            "val_mse": self.val_mse.tolist(),
            "normalized_val_mse": self.normalized_val_mse.tolist(),
            "cross_correlation": self.cross_correlation.tolist(),
            "time": self.time.tolist()
        }

    def toJSON(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=4)