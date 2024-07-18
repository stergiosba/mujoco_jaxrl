from Environment import Env
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import mujoco.viewer
import mujoco.mjx as mjx
import time
import jax
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

model_path = "./google_barkour_vb/scene_mjx.xml"

key = jr.key(0)
options = {"timestep": 1 / 60, "integrator": 0, "gravity": [0, 0, -9.81]}
env = Env(model_path, options=options)
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

state = env.reset(key)
action = jnp.ones((1, 12))
print("JIT-compiling the model physics step...")
start = time.time()
jitted_step = eqx.filter_jit(env.step).lower(state, action).compile()
elapsed = time.time() - start
print(elapsed)

state = env.reset(key)
@jax.jit
def standing_controller(time):
    rate = 1
    action = (
        rate
        * time
        / T
        * jnp.array(
            [
                [
                    0.0284,
                    0.216,
                    0.952,
                    -0.218,
                    -0.112,
                    0.851,
                    -0.199,
                    0.172,
                    0.851,
                    -0.369,
                    -0.0248,
                    0.762,
                ]
            ]
        )
    )
    return action


start = time.time()
t = 0
T = 1023
with mujoco.viewer.launch_passive(m, d) as v:
    while t < T:
        action = standing_controller(t)
        state = jitted_step(state, action)
        mjx.get_data_into(d, m, state.dx)
        v.sync()

        elapsed = time.time() - start
        if elapsed < m.opt.timestep:
            time.sleep(m.opt.timestep - elapsed)

        t += 1
print("End")
