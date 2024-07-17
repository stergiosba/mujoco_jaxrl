from Environment import Env
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import mujoco.viewer
import mujoco.mjx as mjx
import time
import jax

model_path = "./model.xml"

key = jr.key(0)
options = {"timestep": 1 / 60, "integrator": 1}
env = Env(model_path, options=options)
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

state = env.reset(key)
action = jnp.ones((1, 2))
print('JIT-compiling the model physics step...')
start = time.time()
jitted_step = eqx.filter_jit(env.step).lower(state, action).compile()
elapsed = time.time() - start
print(elapsed)


start = time.time()
t = 0
T = 1023
with mujoco.viewer.launch_passive(m, d) as v:
    while t<T:
        state = jitted_step(state, action)
        mjx.get_data_into(d, m, state.dx)
        v.sync()

        elapsed = time.time() - start
        if elapsed < m.opt.timestep:
            time.sleep(m.opt.timestep - elapsed)

        t+=1
print("End")
