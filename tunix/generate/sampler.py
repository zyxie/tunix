# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vanilla sampler for LLM generation."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any
from typing import Optional

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
from tunix.generate import utils
import tunix.generate.beam_search as beam_search_lib
import tunix.generate.tokenizer_adapter as tok_adapter

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]

# Constants
_DEFAULT_TEMPERATURE = 0.0
_SUPPORTED_SAMPLING_MODES = {'greedy', 'top_p', 'beam_search'}


@flax.struct.dataclass
class _BufferState:
  """State for token and logits buffers."""
  token_buffer: jnp.ndarray  # [B, L]
  positions: jnp.ndarray  # [B, L]
  logits_buffer: jnp.ndarray | None  # [B, L, V]


@flax.struct.dataclass  
class _SamplingConfig:
  """Configuration for sampling behavior."""
  sampling_mode: str = flax.struct.field(pytree_node=False)
  temperature: float = flax.struct.field(pytree_node=False)
  sampling_parameters: dict[str, float | int] = flax.struct.field(pytree_node=False)
  forbidden_token_ids: Sequence[int] | None = flax.struct.field(pytree_node=False)
  seed: jax.Array


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""
  decoding_step: jnp.int32
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)
  total_sampling_steps: int = flax.struct.field(pytree_node=False)
  done: jnp.ndarray  # [B]
  cache: dict[str, dict[str, jaxtyping.Array]]
  
  # Nested state objects
  buffer_state: _BufferState
  sampling_config: _SamplingConfig
  beam_search_sampling_state: beam_search_lib._BeamSearchSamplingState | None = None


@dataclasses.dataclass
class SamplerOutput:
  """Output of the sampler."""

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[jax.Array] | jax.Array

  # Tokens corresponding to the generated samples.
  tokens: list[jax.Array] | jax.Array

  # Left padded prompt tokens.
  padded_prompt_tokens: jax.Array


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


def _sample_top_p(
    probs: jnp.ndarray, p: float, key: jax.Array, k: Optional[int] = None
) -> jnp.ndarray:
  """Sample a token using top-p sampling."""
  k = probs.shape[-1] if k is None else k
  probs_sorted, indices = jax.lax.top_k(probs, k=k)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > p
  probs_sorted = jnp.where(mask, 0.0, probs_sorted)
  probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

  next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

  next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)
  return next_token


def sample_top_p(
    logits, key, temperature: float, top_p: float, top_k: Optional[int]
):
  probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
  next_token = _sample_top_p(probs, top_p, key, top_k)
  return next_token


def sample_best(logits):
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  return next_token


def _init_cache(
    n_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> Cache:
  """Create KV cache for the transformer.

  Args:
    n_layers: The number of attention layers.
    cache_size: The size of the cache.
    batch_size: The batch size.
    num_kv_heads: The number of KV attention heads.
    head_dim: The dimension of the KV attention head.
    dtype: The data type of the cache.

  Returns:
    The KV cache for one attention block.
  """

  def _init_layer_cache() -> LayerCache:
    return {
        'k': jnp.zeros(
            (batch_size, cache_size, num_kv_heads, head_dim), dtype=dtype
        ),
        'v': jnp.zeros(
            (batch_size, cache_size, num_kv_heads, head_dim), dtype=dtype
        ),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
    }

  cache = {f'layer_{i}': _init_layer_cache() for i in range(n_layers)}
  return cache


class Sampler:
  """Sampler for transformer model."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: CacheConfig,
  ):
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      cache_config: configuration for the KV cache.
    """
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.cache_config = cache_config
    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    # we separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time
    self._compiled_decode_fn = jax.jit(self._decode_fn)
    self._compiled_prefill_fn = jax.jit(self._prefill_fn)

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> None:

    def get_all_param_types(tree):
      param_types = set()
      jax.tree_util.tree_map(
          lambda x: param_types.add(type(x)),
          tree,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      return param_types

    def check_tree_structure(tree1, tree2):
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            'New state must have the same structure as the old state.'
            f' {jax.tree_util.tree_structure(tree1)} vs'
            f' {jax.tree_util.tree_structure(tree2)}'
        )

      def check_shape_dtype_sharding(x, y):
        return (
            jnp.shape(x) == jnp.shape(y)
            and jnp.dtype(x) == jnp.dtype(y)
            and x.sharding == y.sharding
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            f' state. {tree1} vs {tree2}'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      # Full state replacement.
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      # LoRA state replacement.
      assert (
          len(param_types) == 1 and nnx.LoRAParam in param_types
      ), f'Only LoRAParam is supported. Invalid: {param_types}'
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  @property
  def dtype(self) -> jnp.dtype:
    return self._flattened_transformer_state[0].dtype

  def _create_sampling_config(
      self,
      temperature: float,
      top_p: Optional[float],
      top_k: Optional[int],
      beam_size: Optional[int],
      forbidden_token_ids: Sequence[int] | None,
      seed: jax.Array,
  ) -> _SamplingConfig:
    """Creates sampling configuration."""
    sampling_parameters = {}
    sampling_mode = [None]

    if beam_size is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'beam_search')
      sampling_parameters['beam_size'] = beam_size

    if top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')
      sampling_parameters['top_p'] = top_p
      sampling_parameters['top_k'] = top_k

    if sampling_mode[0] is None:
      sampling_mode[0] = 'greedy'

    logging.debug('Using sampling mode: %s', sampling_mode[0])
    
    return _SamplingConfig(
        sampling_mode=sampling_mode[0],
        temperature=temperature,
        sampling_parameters=sampling_parameters,
        forbidden_token_ids=forbidden_token_ids,
        seed=seed,
    )

  def _create_buffer_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
  ) -> _BufferState:
    """Creates buffer state for tokens and logits."""
    batch_size, num_input_tokens = all_input_ids.shape
    buffer_size = total_sampling_steps + 1

    token_buffer = jnp.full(
        (batch_size, buffer_size),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    input_mask = input_mask.at[:, :num_input_tokens].set(
        all_input_ids != self.tokenizer.pad_id()
    )
    positions = utils.build_positions_from_mask(input_mask)

    logits_buffer = None
    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, buffer_size, self.transformer.num_embed),
          dtype=jnp.float32,
      )

    return _BufferState(
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
    )

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
      forbidden_token_ids: Sequence[int] | None,
      temperature: float,
      top_p: Optional[float],
      top_k: Optional[int],
      seed: jax.Array,
      beam_size: Optional[int],
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size, num_input_tokens = all_input_ids.shape

    buffer_state = self._create_buffer_state(all_input_ids, total_sampling_steps, include_logits)
    sampling_config = self._create_sampling_config(
        temperature, top_p, top_k, beam_size, forbidden_token_ids, seed
    )

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)
    cache = _init_cache(
        n_layers=self.cache_config.num_layers,
        cache_size=self.cache_config.cache_size,
        batch_size=batch_size,
        num_kv_heads=self.cache_config.num_kv_heads,
        head_dim=self.cache_config.head_dim,
        dtype=self.dtype,
    )

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        total_sampling_steps=total_sampling_steps,
        done=done,
        cache=cache,
        buffer_state=buffer_state,
        sampling_config=sampling_config,
        beam_search_sampling_state=None,
    )

  def tokenize(self, input_string: str) -> jax.Array:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = jnp.array(bos_tok + input_ids, dtype=jnp.int32)
    return input_ids

  def _apply_forbidden_tokens(self, logits: jnp.ndarray, forbidden_token_ids: Sequence[int] | None) -> jnp.ndarray:
    """Applies forbidden token constraints to logits."""
    if forbidden_token_ids:
      logits = logits.at[:, :, forbidden_token_ids].set(-jnp.inf)
    return logits

  def _sample_token(
      self,
      logits: jnp.ndarray,
      sampling_config: _SamplingConfig,
      decoding_step: jnp.int32,
  ) -> jnp.ndarray:
    """Samples next token based on sampling configuration."""
    if sampling_config.sampling_mode == 'greedy':
      return sample_best(logits)
    elif sampling_config.sampling_mode == 'top_p':
      key = jax.random.fold_in(sampling_config.seed, decoding_step)
      return sample_top_p(
          logits,
          key,
          sampling_config.temperature,
          sampling_config.sampling_parameters['top_p'],
          sampling_config.sampling_parameters['top_k'],
      )
    else:
      raise ValueError(f'Unsupported sampling mode: {sampling_config.sampling_mode}')

  def _sample(
      self,
      logits: jnp.ndarray,
      eos: int,
      cache: dict[str, dict[str, jaxtyping.Array]],
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""
    logits = logits[:, -1][:, None, :]  # B, 1, V
    logits = self._apply_forbidden_tokens(logits, sampler_state.sampling_config.forbidden_token_ids)
    
    decoding_step = sampler_state.decoding_step
    token_buffer = sampler_state.buffer_state.token_buffer
    done = sampler_state.done
    logits_buffer = sampler_state.buffer_state.logits_buffer
    beam_search_state = sampler_state.beam_search_sampling_state

    if sampler_state.sampling_config.sampling_mode == 'beam_search':
      beam_search_state, updated_args = beam_search_lib.beam_search_step(
          logits=logits,
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=logits_buffer,
          state=beam_search_state,
          pad_token_id=eos,
          decoding_step=decoding_step,
      )
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      logits_buffer = updated_args['logits_buffer']
    else:
      next_token_candidate = self._sample_token(logits, sampler_state.sampling_config, decoding_step)
      token_buffer = token_buffer.at[:, decoding_step + 1].set(next_token_candidate)

    done = done | jnp.equal(token_buffer[:, decoding_step + 1], eos)
    
    updated_buffer_state = _BufferState(
        token_buffer=token_buffer,
        positions=sampler_state.buffer_state.positions,
        logits_buffer=logits_buffer,
    )
    
    return _SamplingState(
        decoding_step=decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        total_sampling_steps=sampler_state.total_sampling_steps,
        done=done,
        cache=cache,
        buffer_state=updated_buffer_state,
        sampling_config=sampler_state.sampling_config,
        beam_search_sampling_state=beam_search_state,
    )

  def _prefill_fn(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs prefill."""
    batch_size = sampler_state.buffer_state.token_buffer.shape[0]

    tokens = jax.lax.dynamic_slice(
        sampler_state.buffer_state.token_buffer,
        start_indices=jnp.zeros(
            (sampler_state.buffer_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )
    step_positions = jax.lax.dynamic_slice(
        sampler_state.buffer_state.positions,
        start_indices=jnp.zeros(
            (sampler_state.buffer_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )

    input_mask = tokens != self.tokenizer.pad_id()
    attention_mask = utils.make_causal_attn_mask(
        input_mask, self.cache_config.cache_size
    )

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    token_buffer = sampler_state.buffer_state.token_buffer
    done = sampler_state.done
    positions = sampler_state.buffer_state.positions
    beam_search_sampling_state = None
    if sampler_state.buffer_state.logits_buffer is not None:
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.buffer_state.logits_buffer,
          logits.astype(sampler_state.buffer_state.logits_buffer.dtype),
          (0, 1, 0),
      )
    else:
      logits_buffer = sampler_state.buffer_state.logits_buffer

    if sampler_state.sampling_config.sampling_mode == 'beam_search':
      # init beam state in prefill instead of init as one minor optimization
      # to avoid running unnecessary prefill for
      # duplicated input prompt per beam.
      sampling_state, updated_args = beam_search_lib.init_batched_beam_state(
          logits=logits,
          input_token_buffer=sampler_state.buffer_state.token_buffer,
          initial_cache=cache,
          done=sampler_state.done,
          positions=sampler_state.buffer_state.positions,
          logits_buffer=sampler_state.buffer_state.logits_buffer,
          beam_size=int(sampler_state.sampling_config.sampling_parameters['beam_size']),
      )
      beam_search_sampling_state = sampling_state
      logits = updated_args['logits']
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      positions = updated_args['positions']
      logits_buffer = updated_args['logits_buffer']

    updated_buffer_state = _BufferState(
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
    )
    
    updated_sampling_state = _SamplingState(
        decoding_step=sampler_state.decoding_step,
        num_input_tokens=sampler_state.num_input_tokens,
        total_sampling_steps=sampler_state.total_sampling_steps,
        done=done,
        cache=cache,
        buffer_state=updated_buffer_state,
        sampling_config=sampler_state.sampling_config,
        beam_search_sampling_state=beam_search_sampling_state,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.tokenizer.eos_id(),
        sampler_state=updated_sampling_state,
    )
    return updated_sampler_state

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs a single sampling step."""
    batch_size = sampler_state.buffer_state.token_buffer.shape[0]
    decoding_step = sampler_state.decoding_step

    last_token = sampler_state.buffer_state.token_buffer[:, decoding_step]
    last_token = last_token.reshape((batch_size, 1))
    step_positions = jnp.expand_dims(
        sampler_state.buffer_state.positions[:, decoding_step], -1
    )

    input_mask = sampler_state.buffer_state.token_buffer == self.tokenizer.pad_id()
    attention_mask = utils.compute_attention_masks(
        decoding_step, self.cache_config.cache_size, input_mask
    )

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        last_token,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.tokenizer.eos_id(),
        sampler_state=sampler_state,
    )

    if updated_sampler_state.buffer_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = updated_sampler_state.buffer_state.logits_buffer.at[
          :, decoding_step + 1
      ].set(next_logits)
      updated_buffer_state = dataclasses.replace(
          updated_sampler_state.buffer_state,
          logits_buffer=logits_buffer,
      )
      updated_sampler_state = dataclasses.replace(
          updated_sampler_state,
          buffer_state=updated_buffer_state,
      )
    return updated_sampler_state

  def __call__(
      self,
      input_strings: Sequence[str],
      total_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      forbidden_tokens: Sequence[str] | None = None,
      temperature: float = _DEFAULT_TEMPERATURE,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: jax.Array | None = None,
      pad_output: bool = False,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    If top_p is provided, the sampling mode will be top_p.
    If beam_size is provided, the sampling mode will be beam_search.
    If None of them are provided, the sampling mode will be greedy.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      max_prompt_length: maximum length of the prompt. Specify to avoid
        recompilation on different prompt lengths.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      forbidden_tokens: list of tokens that are forbidden to be generated. Each
        token must map to a single token id in the vocab.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      top_k: top-k sampling threshold.
      beam_size: beam size for beam search.
      seed: random seed for sampling.
      pad_output: whether to pad the output to maximum length. If this set as
        True, the output len will be total_generation_steps if echo is False,
        otherwise it will be total_generation_steps + max_prompt_length. The
        padding now only supports right padding. Can modify to support left
        padding if needed.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    self._validate_generation_params(total_generation_steps, temperature)
    forbidden_token_ids = self._process_forbidden_tokens(forbidden_tokens)

    tokens = [self.tokenize(x) for x in input_strings]
    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)
    all_input_ids = jnp.array([
        utils.pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in tokens
    ])
    total_sampling_steps = max_prompt_length + total_generation_steps
    if total_sampling_steps > self.cache_config.cache_size:
      raise ValueError(
          'Total sampling steps must be less than the cache size'
          f' {self.cache_config.cache_size}.'
      )

    if seed is None:
      seed = jax.random.PRNGKey(0)
    sampling_state = self.init_sample_state(
        all_input_ids,
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        beam_size=beam_size,
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state, sampling_state
    )

    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )

    token_buffers = sampling_state.buffer_state.token_buffer
    logits_buffers = sampling_state.buffer_state.logits_buffer

    if sampling_state.sampling_config.sampling_mode == 'beam_search':
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.buffer_state.token_buffer,
          sampling_state.buffer_state.logits_buffer,
      )
      token_buffers = updated_args['token_buffer']
      logits_buffers = updated_args['logits_buffer']
      # delete the sampling state in case the further referece
      # if need more internal states, they should be updated by
      # finalize_beam_search_state
      del sampling_state
    if pad_output:
      max_len = (total_sampling_steps if echo else total_generation_steps) + 1
      lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
          token_buffers,
          logits_buffers,
          return_logits,
          echo,
          self.tokenizer.pad_id(),
          self.tokenizer.eos_id(),
          max_prompt_length,
          max_len,
      )
      decoded_outputs = [
          self.tokenizer.decode(tokens[:length].tolist())
          for tokens, length in zip(out_tokens, lengths)
      ]
    else:
      out_tokens = []
      out_logits = []
      lengths = []
      for i, token_buffer in enumerate(token_buffers):
        start_idx = (
            utils.find_first_non_pad_idx(token_buffer, self.tokenizer.pad_id())
            if echo
            else max_prompt_length
        )
        end_idx = (
            utils.find_first_eos_idx(
                token_buffer[max_prompt_length:], self.tokenizer.eos_id()
            )
            + max_prompt_length
        )
        out_tokens.append(token_buffer[start_idx:end_idx])
        if return_logits:
          out_logits.append(logits_buffers[i][start_idx:end_idx])
        lengths.append(end_idx - start_idx)

      decoded_outputs = [
          self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
      ]

    result = SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
    )
    return result

  def _validate_generation_params(self, total_generation_steps: int, temperature: float) -> None:
    """Validates generation parameters."""
    if total_generation_steps <= 0:
      raise ValueError(f'total_generation_steps must be positive, got {total_generation_steps}')
    if temperature < 0:
      raise ValueError(f'temperature must be non-negative, got {temperature}')

  def _process_forbidden_tokens(self, forbidden_tokens: Sequence[str] | None) -> tuple[int, ...] | None:
    """Processes forbidden tokens into token IDs."""
    if forbidden_tokens is None:
      return None
    
    forbidden_token_ids = []
    for token in forbidden_tokens:
      token_id = self.tokenizer.encode(token)
      if len(token_id) != 1:
        raise ValueError(
            'Forbidden tokens must map to single token ids in the vocab.'
        )
      forbidden_token_ids.extend(token_id)
    return tuple(forbidden_token_ids)
