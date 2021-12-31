"""A differentiable machine that understands `DUP`, `ADD`, `STOP`.

The state of the machine comprises a data stack, a program counter and whether the machine has halted.

The parameters of the machine are the lines of code.

The learning task is to learn multiplication by d modulo n,
that is f(x)=(x*d) % n.
For n=5 and  d=2, we generate the examples
f(0) = 0, f(1) = 2, f(2) = 4, f(3) = 1, f(4) = 3
and the machine learns the program
DUP, ADD, STOP, *
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

N = flags.DEFINE_integer('n', 5, 'uniformly, number of integers, number of lines of code and of data stack size')
D = flags.DEFINE_integer('d', 2, 'learn f(x)=(x*d)%n')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

def instruction_names():
    return ['DUP', 'ADD', 'STOP']

class Machine(hk.RNNCore):
    def __init__(
            self,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.n = N.value
        self.stop_matrix = jnp.identity(self.n)
        self.inc_matrix =  jnp.roll(jnp.identity(self.n), 1, axis=1)
        self.dec_matrix = jnp.roll(jnp.identity(self.n), -1, axis=1)
        self.inames = instruction_names()
        self.ni = len(self.inames)

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        data_p = self.state_data_p(new_state)
        data = self.state_data(new_state)
        new_data_value = self.read_value(data_p, data)
        new_halted = self.state_halted(new_state)
        return new_state, new_state

    def read_value(self, data_p, data):
        return jnp.matmul(data_p.T, data).T

    def write_value(self, data_p, data, data_value):
        old = jnp.outer(data_p, jnp.ones(self.n)) * data
        new = jnp.outer(data_p, data_value)
        return data - old + new

    def is_instr(self, iname, index):
        return self.inames[index] == iname

    def execute_instr(self, i, data_p, data, pc):
        if self.is_instr('STOP', i):
            return (data_p, data, pc)
        next_pc = jnp.matmul(pc, self.inc_matrix)
        if self.is_instr('DUP', i):
            data_value = self.read_value(data_p, data)
            next_data_p = jnp.matmul(data_p, self.inc_matrix)
            next_data = self.write_value(next_data_p, data, data_value)
            return (next_data_p, next_data, next_pc)
        elif self.is_instr('ADD', i):
            n1 = self.read_value(data_p, data)
            next_data_p = jnp.matmul(data_p, self.dec_matrix)
            n2 = self.read_value(next_data_p, data)
            nr = self.add(n1, n2)
            next_data = self.write_value(next_data_p, data, nr)
            return (next_data_p, next_data, next_pc)
        # let an error be...

    def add(self, n1, n2):
        nr = jnp.zeros(self.n)
        m = jnp.identity(self.n)
        for i in range(self.n):
            nr += n2[i] * jnp.matmul(n1, m)
            m = jnp.matmul(m, self.inc_matrix)
        return nr

    def initial_state(self, batch_size: Optional[int]):
        data_p = jnp.zeros(self.n).at[0].set(1)
        data = jnp.zeros([self.n, self.n])
        for i in range(self.n):
            data = data.at[(i,0)].set(1)
        data = jnp.reshape(data, self.n*self.n)
        pc = jnp.zeros([self.n]).at[0].set(1)
        halted = jnp.array([0, 1])
        state = jnp.concatenate((data_p, data, pc, halted))
        assert batch_size is None
        return state

    def sm(self, x):
        return jax.nn.softmax(SOFTMAX_SHARP.value*x)

    def state_data_p(self, state):
        return state[0:self.n]

    def state_data(self, state):
        data = state[self.n:self.n*self.n+self.n]
        return jnp.reshape(data, [self.n, self.n])

    def state_pc(self, state):
        return state[self.n*self.n+self.n:self.n*self.n+2*self.n]

    def state_halted(self, state):
        return state[self.n*self.n+2*self.n:self.n*self.n+2*self.n+2]

    def step(self, state):
        code = self.get_code()
        data_p = self.sm(self.state_data_p(state))
        data = self.state_data(state)
        for i in range(self.n):
            data = data.at[i].set(self.sm(data[i]))
        pc = self.sm(self.state_pc(state))
        halted = self.sm(self.state_halted(state))
        sel = jnp.zeros(self.ni)
        for i in range(self.n):
            sel += pc[i] * self.sm(code[i])
        new_data_p = jnp.zeros(self.n)
        new_data = jnp.zeros([self.n, self.n])
        new_pc = jnp.zeros(self.n)
        for i in range(self.ni):
            (delta_data_p, delta_data, delta_pc) = self.execute_instr(i, data_p, data, pc)
            new_data_p += sel[i] * delta_data_p
            new_data += sel[i] * delta_data
            new_pc += sel[i] * delta_pc
        next_data_p = halted[0] * data_p + halted[1] * new_data_p
        next_data = halted[0] * data + halted[1] * new_data
        next_pc = halted[0] * pc + halted[1] * new_pc
        halting = 0.0
        not_halting = 0.0
        for i in range(self.ni):
            if self.is_instr('STOP', i):
                halting += sel[i]
            else:
                not_halting += sel[i]
        next_halted = jnp.array([halted[0] + halted[1]*halting, halted[1]*not_halting])
        next_data = jnp.reshape(next_data, self.n*self.n)
        next_state = jnp.concatenate((next_data_p, next_data, next_pc, next_halted))
        return next_state

    def get_code(self):
        return hk.get_parameter('code', [self.n, self.ni], init=self.make_code_fun())

    def make_code_fun(self):
        # all STOPs
        code = jnp.array([[1.0 if self.is_instr('STOP', i) else 0.0 for i in range(self.ni)] for l in range(self.n)])

        def code_fun(shape, dtype):
            return code
        return code_fun

    def load_data(self, state, data_value):
        # assumes the pointer is at the reset
        next_state = state.at[self.n:self.n*2].set(data_value)
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
  states, _ = hk.dynamic_unroll(core, [jnp.zeros(core.n) for i in range(sequence_length)], initial_state)
  return states

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  # Note the [-1] is to consider only the final output, not the intermediary data points.
  (logits, halted) = forward(t['input'])
  log_probs = jax.nn.log_softmax(SOFTMAX_SHARP.value*logits[-1])
  log_probs_halted = jax.nn.log_softmax(SOFTMAX_SHARP.value*halted[-1])

  one_hot_labels = t['target'][-1]
  loss = -jnp.sum(one_hot_labels * log_probs)
  loss -= log_probs_halted[0]
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
    flags.FLAGS([""])

    train_data = itertools.cycle(train_data_inc(D.value))

    code = jnp.zeros([5, 3])
    code = code.at[(0,0)].set(1)
    code = code.at[(1,1)].set(1)
    code = code.at[(2,2)].set(1)
    code = code.at[(3,2)].set(1)
    code = code.at[(4,2)].set(1)

    params = {'machine': {'code': code } }

    print('MACHINE CODE', 'for learning f(x)=(x*%d)%%%d' % (D.value, N.value))
    names = instruction_names()
    print([names[x]for x in to_discrete(params['machine']['code'])])


    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(N.value):
        t = next(train_data)
        outputs = forward_fn(params, t['input'])
        print('input:', jnp.argmax(t['input']).item())
        #print('output steps:', to_discrete(logits))
        print('outputs:', outputs)

if __name__ == '__main__':
    app.run(main)
