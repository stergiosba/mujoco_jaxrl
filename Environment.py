from jaxtyping import Array, Float, PyTree
import mujoco
import mujoco.mjx as mjx
import equinox as eqx
import jax.numpy as jnp
import jax
from typing import Dict
from mujoco.mjx._src.types import Data

class EnvState(eqx.Module):
    dx: Data


class Env(eqx.Module):
    m: mujoco.MjModel
    mx: PyTree

    def __init__(self, model_path: str, options: Dict = None):
        m = mujoco.MjModel.from_xml_path(model_path)

        if options is not None:
            m.opt.timestep = options["timestep"]
            m.opt.integrator = options["integrator"]
        self.m = m
        self.mx = mjx.put_model(m)

    def reset(self, key):
        state = EnvState(dx=mjx.make_data(self.mx))
        return state

    def _step_internal(self, dx, action):

        def body_fn(dx, action):
            dx = dx.replace(ctrl=jnp.array(action))
            new_dx = mjx.step(self.mx, dx)

            return new_dx, None

        carry_out, _ = jax.lax.scan(body_fn, dx, action)

        return carry_out

    def step(self, state, action):
        dx = state.dx
        new_dx = self._step_internal(dx, action)
        new_state = EnvState(dx=new_dx)
        return new_state
