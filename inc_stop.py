"""A differentiable machine that only understands `INC` and `STOP`.

There is no data stack, just a data point.
There is also a program counter and a code bank.

We're using an RNN, where the output is the data point (top of the 'stack').
For an example computation, we have to provide the data point at each step.

The code parameterizes the neural network. The weights represent the
code, and we learn the code.

We initialize the code with all STOPs.
"""

from typing import Any, NamedTuple, Optional

from absl import app
from absl import flags
from absl import logging

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

import itertools

N = flags.DEFINE_integer('n', 5, 'uniform number: of integers, of lines of code, of instructions (with dups)')
D = flags.DEFINE_integer('d', 3, 'learn f(x)=(x+d)%n')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 1000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

class Machine(hk.RNNCore):
    def __init__(
            self,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.n = N.value
        self.stop_matrix = jnp.identity(self.n)
        self.inc_matrix =  jnp.identity(self.n)
        a0 = self.inc_matrix[0]
        for i in range(self.n-1):
            self.inc_matrix = self.inc_matrix.at[i].set(self.inc_matrix[i+1])
        self.inc_matrix = self.inc_matrix.at[self.n-1].set(a0)
        unique_instructions = [self.stop_matrix, self.inc_matrix]
        n_unique = len(unique_instructions)
        assert self.n >= n_unique
        instructions = []
        for i in range(self.n):
            instructions.append(unique_instructions[i%n_unique])
        self.data_instructions = instructions
        self.pc_instructions = instructions

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        new_data = new_state[0:self.n]
        return new_data, new_state

    def initial_state(self, batch_size: Optional[int]):
        data = jnp.zeros([self.n]).at[0].set(1)
        pc = jnp.zeros([self.n]).at[0].set(1)
        state = jnp.concatenate((data, pc))
        assert batch_size is None
        return state

    def step(self, state):
        code = self.get_code()
        data = jax.nn.softmax(SOFTMAX_SHARP.value*state[0:self.n])
        pc = jax.nn.softmax(SOFTMAX_SHARP.value*state[self.n:2*self.n])
        sel = jnp.zeros(self.n)
        for i in range(self.n):
            sel += pc[i] * jax.nn.softmax(SOFTMAX_SHARP.value*code[i])
        data_instr = jnp.reshape(jnp.zeros(self.n*self.n), (self.n, self.n))
        pc_instr = jnp.reshape(jnp.zeros(self.n*self.n), (self.n, self.n))
        for i in range(self.n):
            data_instr += sel[i] * self.data_instructions[i]
            pc_instr += sel[i] * self.pc_instructions[i]
        next_data = jnp.matmul(data, data_instr)
        next_pc = jnp.matmul(pc, pc_instr)
        next_state = jnp.concatenate((next_data, next_pc))
        return next_state

    def get_code(self):
        return hk.get_parameter('code', [self.n, self.n], init=self.make_code_fun())

    def make_code_fun(self):
        code = jnp.array([[1.0 if i==0 else 0.0 for i in range(self.n)] for l in range(self.n)])
        def code_fun(shape, dtype):
            return code
        return code_fun

    def load_data(self, state, data):
        next_state = state.at[0:self.n].set(data)
        return next_state

class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState

def make_network() -> Machine:
    model = Machine()
    return model

def make_optimizer() -> optax.GradientTransformation:
  """Defines the optimizer."""
  return optax.adam(LEARNING_RATE.value)

def forward(input) -> jnp.ndarray:
  core = make_network()
  sequence_length = core.n
  initial_state = core.initial_state(batch_size=None)
  initial_state = core.load_data(initial_state, input)
  logits, _ = hk.dynamic_unroll(core, [jnp.zeros(5) for i in range(sequence_length)], initial_state)
  return logits

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  # Note: this function is impure; we hk.transform() it below.
  sequence_length = N.value
  logits = forward(t['input'])
  log_probs = jax.nn.log_softmax(logits)
  one_hot_labels = t['target']
  loss = -jnp.sum(one_hot_labels * log_probs) / sequence_length
  return loss

@jax.jit
def update(state: TrainingState, t) -> TrainingState:
  """Does a step of SGD given inputs & targets."""
  _, optimizer = make_optimizer()
  _, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
  gradients = jax.grad(loss_fn)(state.params, t)
  updates, new_opt_state = optimizer(gradients, state.opt_state)
  new_params = optax.apply_updates(state.params, updates)
  return TrainingState(params=new_params, opt_state=new_opt_state)

def train_data_inc(d):
    r = []
    for i in range(N.value):
        data = jax.nn.one_hot(i, N.value)
        target = jax.nn.one_hot([(i+j+1)%N.value if j < d else (i+d)%N.value for j in range(N.value)], N.value)
        r.append({'input':data, 'target':target})
    return r

def main(_):
    #flags.FLAGS([""])

    train_data = itertools.cycle(train_data_inc(D.value))

    params_init, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
    opt_init, _ = make_optimizer()

    loss_fn = jax.jit(loss_fn)

    rng = hk.PRNGSequence(SEED.value)
    initial_params = params_init(next(rng), next(train_data))
    initial_opt_state = opt_init(initial_params)
    state = TrainingState(params=initial_params, opt_state=initial_opt_state)

    for step in range(TRAINING_STEPS.value + 1):
        t = next(train_data)
        state = update(state, t)

    print(state.params['machine']['code'])

    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(N.value):
        t = next(train_data)
        print('input :', t['input'])
        print(forward_fn(state.params, t['input']))

if __name__ == '__main__':
    app.run(main)
