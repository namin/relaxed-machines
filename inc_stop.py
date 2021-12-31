"""A differentiable machine that only understands `INC` and `STOP`.

There is no data stack, just a data point.
There is also a program counter and a code bank.

The code parameterizes the neural network. The parameters represent the
code, and we learn the code.

We initialize the code with all STOPs.

The learning task is to learn counting (modulo n) by incrementing d times.
For n=5 and  d=3, we generate the examples
f(0) = 3, f(1) = 4, f(2) = 0, f(3) = 1, f(4) = 2
and the machine learns the program
INC, INC, INC, STOP, ...garbage...
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

N = flags.DEFINE_integer('n', 5, 'uniformly, number of integers and number of lines of code')
D = flags.DEFINE_integer('d', 3, 'learn f(x)=(x+d)%n')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100000, '')
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
        instructions = [self.stop_matrix, self.inc_matrix]
        self.ni = len(instructions)
        self.data_instructions = instructions
        self.pc_instructions = instructions

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        new_data = new_state[0:self.n]
        return new_data, new_state

    def initial_state(self, batch_size: Optional[int]):
        data = jnp.zeros([self.n]).at[0].set(1)
        pc = jnp.zeros([self.n]).at[0].set(1)
        halted = jnp.array([0, 1])
        state = jnp.concatenate((data, pc, halted))
        assert batch_size is None
        return state

    def step(self, state):
        code = self.get_code()
        data = jax.nn.softmax(SOFTMAX_SHARP.value*state[0:self.n])
        pc = jax.nn.softmax(SOFTMAX_SHARP.value*state[self.n:2*self.n])
        halted = jax.nn.softmax(SOFTMAX_SHARP.value*state[2*self.n:2*self.n+2])
        sel = jnp.zeros(self.ni)
        for i in range(self.n):
            sel += pc[i] * jax.nn.softmax(SOFTMAX_SHARP.value*code[i])
        data_instr = jnp.reshape(jnp.zeros(self.n*self.n), (self.n, self.n))
        pc_instr = jnp.reshape(jnp.zeros(self.n*self.n), (self.n, self.n))
        for i in range(self.ni):
            data_instr += sel[i] * self.data_instructions[i]
            pc_instr += sel[i] * self.pc_instructions[i]
        next_data = halted[0] * data + halted[1] * jnp.matmul(data, data_instr)
        next_pc = halted[0] * pc + halted[1] * jnp.matmul(pc, pc_instr)
        next_halted = jnp.array([halted[0] + halted[1]*sel[0], halted[1]*sel[1]])
        next_state = jnp.concatenate((next_data, next_pc, next_halted))
        return next_state

    def get_code(self):
        return hk.get_parameter('code', [self.n, self.ni], init=self.make_code_fun())

    def make_code_fun(self):
        code = jnp.array([[1.0 if i==0 else 0.0 for i in range(self.ni)] for l in range(self.n)])
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
  # the [-1] is to consider only the final output, not the intermediary data points
  logits = forward(t['input'])[-1]
  log_probs = jax.nn.log_softmax(logits)
  one_hot_labels = t['target'][-1]
  loss = -jnp.sum(one_hot_labels * log_probs)
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

def to_discrete(a):
    return [jnp.argmax(x).item() for x in a]

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

    #print(state.params['machine']['code'])
    print('MACHINE CODE')
    print(to_discrete(state.params['machine']['code']))

    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(N.value):
        t = next(train_data)
        logits = forward_fn(state.params, t['input'])
        #print('input:', t['input'])
        #print(logits)
        print('input:', jnp.argmax(t['input']).item())
        print('output steps:', to_discrete(logits))

if __name__ == '__main__':
    app.run(main)
