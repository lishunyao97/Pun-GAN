# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

PUNGAN_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    "BeamSearchDecoderOutput",
    "BeamSearchDecoderState",
    "BeamSearchDecoder",
    "FinalBeamSearchDecoderOutput",
    "tile_batch",
]


class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState", ("cell_state", "log_probs",
                                                      "finished", "lengths"))):
  pass


class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput",
                           ("scores", "predicted_ids", "parent_ids"))):
  pass


class FinalBeamSearchDecoderOutput(
    collections.namedtuple("FinalBeamDecoderOutput",
                           ["predicted_ids", "beam_search_decoder_output"])):
  """Final outputs returned by the beam search after all decoding is finished.

  Args:
    predicted_ids: The final prediction. A tensor of shape
      `[T, batch_size, beam_width]`.
    beam_search_decoder_output: An instance of `BeamSearchDecoderOutput` that
      describes the state of the beam search.
  """
  pass


def _tile_batch(t, multiplier):
  """Core single-tensor implementation of tile_batch."""
  t = ops.convert_to_tensor(t, name="t")
  shape_t = array_ops.shape(t)
  if t.shape.ndims is None or t.shape.ndims < 1:
    raise ValueError("t must have statically known rank")
  tiling = [1] * (t.shape.ndims + 1)
  tiling[1] = 5
  tiled_static_batch_size = (
      t.shape[0].value * multiplier if t.shape[0].value is not None else None)
  tiled = array_ops.tile(array_ops.expand_dims(t, 1), tiling)
  tiled = tf.concat((tiled,tiled),0)
  tiled = array_ops.reshape(
      tiled, array_ops.concat(([shape_t[0] * multiplier], shape_t[1:]), 0))
  tiled.set_shape(
      tensor_shape.TensorShape(
          [tiled_static_batch_size]).concatenate(t.shape[1:]))
  return tiled


def tile_batch(t, multiplier, name=None):
  """Tile the batch dimension of a (possibly nested structure of) tensor(s) t.

  For each tensor t in a (possibly nested structure) of tensors,
  this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed of
  minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
  `[batch_size * multiplier, s0, s1, ...]` composed of minibatch entries
  `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
  `multiplier` times.

  Args:
    t: `Tensor` shaped `[batch_size, ...]`.
    multiplier: Python int.
    name: Name scope for any created operations.

  Returns:
    A (possibly nested structure of) `Tensor` shaped
    `[batch_size * multiplier, ...]`.

  Raises:
    ValueError: if tensor(s) `t` do not have a statically known rank or
    the rank is < 1.
  """
  flat_t = nest.flatten(t)
  with ops.name_scope(name, "tile_batch", flat_t + [multiplier]):
    return nest.map_structure(lambda t_: _tile_batch(t_, multiplier), t)


def _check_maybe(t):
  if isinstance(t, tensor_array_ops.TensorArray):
    raise TypeError(
        "TensorArray state is not supported by BeamSearchDecoder: %s" % t.name)
  if t.shape.ndims is None:
    raise ValueError(
        "Expected tensor (%s) to have known rank, but ndims == None." % t)


class BeamSearchDecoder(decoder.Decoder):
  """BeamSearch sampling decoder.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```
  """

  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               length_penalty_weight=0.0):
    """Initialize the BeamSearchDecoder.

    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`,
        or `output_layer` is not an instance of `tf.layers.Layer`.
      ValueError: If `start_tokens` is not a vector or
        `end_token` is not a scalar.
    """
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._output_layer = output_layer

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))
    print ('self._embedding_fn',self._embedding_fn)
    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.int32, name="start_tokens")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    def get_embedding(embedding_fn):
        a=embedding_fn(0)
        return a
    a=get_embedding(self._embedding_fn)
    print ("test_tensor a",a)
    self._batch_size = array_ops.size(start_tokens)
    self._beam_width = beam_width
    self._length_penalty_weight = length_penalty_weight
    self._initial_cell_state = nest.map_structure(
        self._maybe_split_batch_beams,
        initial_state, self._cell.state_size)
    self._start_tokens = array_ops.tile(
        array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width])
    self._start_inputs = self._embedding_fn(self._start_tokens)
    self._finished = array_ops.zeros(
        [self._batch_size, self._beam_width], dtype=dtypes.bool)

  @property
  def batch_size(self):
    return self._batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BeamSearchDecoderOutput(
        scores=tensor_shape.TensorShape([self._beam_width]),
        predicted_ids=tensor_shape.TensorShape([self._beam_width]),
        parent_ids=tensor_shape.TensorShape([self._beam_width]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_cell_state)[0].dtype
    return BeamSearchDecoderOutput(
        scores=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        predicted_ids=dtypes.int32,
        parent_ids=dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, start_inputs, initial_state)`.
    """
    finished, start_inputs = self._finished, self._start_inputs
    print ("finished",finished)
    print ("start_inputs",start_inputs)
    initial_state = BeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=array_ops.zeros(
            [self._batch_size, self._beam_width],
            dtype=nest.flatten(self._initial_cell_state)[0].dtype),
        finished=finished,
        lengths=array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.int64))

    return (finished, start_inputs, initial_state)

  def finalize(self, outputs, final_state, sequence_lengths):
    """Finalize and return the predicted_ids.

    Args:
      outputs: An instance of BeamSearchDecoderOutput.
      final_state: An instance of BeamSearchDecoderState. Passed through to the
        output.
      sequence_lengths: An `int64` tensor shaped `[batch_size, beam_width]`.
        The sequence lengths determined for each beam during decode.

    Returns:
      outputs: An instance of FinalBeamSearchDecoderOutput where the
        predicted_ids are the result of calling _gather_tree.
      final_state: The same input instance of BeamSearchDecoderState.
    """
    predicted_ids = beam_search_ops.gather_tree(
        outputs.predicted_ids, outputs.parent_ids,
        sequence_length=sequence_lengths)
    outputs = FinalBeamSearchDecoderOutput(
        beam_search_decoder_output=outputs, predicted_ids=predicted_ids)
    return outputs, final_state

  def _merge_batch_beams(self, t, s=None):
    """Merges the tensor from a batch of beams into a batch by beams.

    More exactly, t is a tensor of dimension [batch_size, beam_width, s]. We
    reshape this into [batch_size*beam_width, s]

    Args:
      t: Tensor of dimension [batch_size, beam_width, s]
      s: (Possibly known) depth shape.

    Returns:
      A reshaped version of t with dimension [batch_size * beam_width, s].
    """
    if isinstance(s, ops.Tensor):
      s = tensor_shape.as_shape(tensor_util.constant_value(s))
    else:
      s = tensor_shape.TensorShape(s)
    t_shape = array_ops.shape(t)
    static_batch_size = tensor_util.constant_value(self._batch_size)
    batch_size_beam_width = (
        None if static_batch_size is None
        else static_batch_size * self._beam_width)
    reshaped_t = array_ops.reshape(
        t, array_ops.concat(
            ([self._batch_size * self._beam_width], t_shape[2:]), 0))
    reshaped_t.set_shape(
        (tensor_shape.TensorShape([batch_size_beam_width]).concatenate(s)))
    return reshaped_t

  def _split_batch_beams(self, t, s=None):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, t is a tensor of dimension [batch_size*beam_width, s]. We
    reshape this into [batch_size, beam_width, s]

    Args:
      t: Tensor of dimension [batch_size*beam_width, s].
      s: (Possibly known) depth shape.

    Returns:
      A reshaped version of t with dimension [batch_size, beam_width, s].

    Raises:
      ValueError: If, after reshaping, the new tensor is not shaped
        `[batch_size, beam_width, s]` (assuming batch_size and beam_width
        are known statically).
    """
    if isinstance(s, ops.Tensor):
      s = tensor_shape.TensorShape(tensor_util.constant_value(s))
    else:
      s = tensor_shape.TensorShape(s)
    t_shape = array_ops.shape(t)
    reshaped_t = array_ops.reshape(
        t, array_ops.concat(
            ([self._batch_size, self._beam_width], t_shape[1:]), 0))
    static_batch_size = tensor_util.constant_value(self._batch_size)
    expected_reshaped_shape = tensor_shape.TensorShape(
        [static_batch_size, self._beam_width]).concatenate(s)
    if not reshaped_t.shape.is_compatible_with(expected_reshaped_shape):
      raise ValueError("Unexpected behavior when reshaping between beam width "
                       "and batch size.  The reshaped tensor has shape: %s.  "
                       "We expected it to have shape "
                       "(batch_size, beam_width, depth) == %s.  Perhaps you "
                       "forgot to create a zero_state with "
                       "batch_size=encoder_batch_size * beam_width?"
                       % (reshaped_t.shape, expected_reshaped_shape))
    reshaped_t.set_shape(expected_reshaped_shape)
    return reshaped_t

  def _maybe_split_batch_beams(self, t, s):
    """Maybe splits the tensor from a batch by beams into a batch of beams.

    We do this so that we can use nest and not run into problems with shapes.

    Args:
      t: Tensor of dimension [batch_size*beam_width, s]
      s: Tensor, Python int, or TensorShape.

    Returns:
      Either a reshaped version of t with dimension
      [batch_size, beam_width, s] if t's first dimension is of size
      batch_size*beam_width or t if not.

    Raises:
      TypeError: If t is an instance of TensorArray.
      ValueError: If the rank of t is not statically known.
    """
    _check_maybe(t)
    if t.shape.ndims >= 1:
      return self._split_batch_beams(t, s)
    else:
      return t

  def _maybe_merge_batch_beams(self, t, s):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, t is a tensor of dimension [batch_size*beam_width, s]. We
    reshape this into [batch_size, beam_width, s]

    Args:
      t: Tensor of dimension [batch_size*beam_width, s]
      s: Tensor, Python int, or TensorShape.

    Returns:
      A reshaped version of t with dimension [batch_size, beam_width, s].

    Raises:
      TypeError: If t is an instance of TensorArray.
      ValueError:  If the rank of t is not statically known.
    """
    _check_maybe(t)
    if t.shape.ndims >= 2:
      return self._merge_batch_beams(t, s)
    else:
      return t

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight

    with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      print ("cell_state",cell_state)
      inputs = nest.map_structure(
          lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
      cell_state = nest.map_structure(
          self._maybe_merge_batch_beams,
          cell_state, self._cell.state_size)
      cell_outputs, next_cell_state = self._cell(inputs, cell_state)
      cell_outputs = nest.map_structure(
          lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
      next_cell_state = nest.map_structure(
          self._maybe_split_batch_beams,
          next_cell_state, self._cell.state_size)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      beam_search_output, beam_search_state = _beam_search_step(
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight)

      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))

    return (beam_search_output, beam_search_state, next_inputs, finished)

def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight):
  """Performs a single step of Beam Search Decoding.

  Args:
    time: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape
      `[batch_size, beam_width, vocab_size]`
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    beam_state: Current state of the beam search.
      An instance of `BeamSearchDecoderState`.
    batch_size: The batch size for this input.
    beam_width: Python int.  The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

  Returns:
    A new beam state.
  """

  static_batch_size = tensor_util.constant_value(batch_size)

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  step_log_probs = nn_ops.log_softmax(logits)
  #step_log_probs",Tensor shape=(?, 10, 56136)
  step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
  #step_log_probs_masked (?, 10, 56136)
  total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs
  #total_probs (?, 10, 56136)
  # Calculate the continuation lengths by adding to all continuing beams.
  vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
  lengths_to_add = array_ops.one_hot(
      indices=array_ops.tile(
          array_ops.reshape(end_token, [1, 1]), [batch_size, beam_width]),
      depth=vocab_size,
      on_value=constant_op.constant(0, dtype=dtypes.int64),
      off_value=constant_op.constant(1, dtype=dtypes.int64),
      dtype=dtypes.int64)
  #lengths_to_add shape=(?, 10, 56136)
  add_mask = (1 - math_ops.to_int64(previously_finished))
  #add_mask shape=(?, 10), dtype=int64
  lengths_to_add = array_ops.expand_dims(add_mask, 2) * lengths_to_add
  #lengths_to_add shape=(?, 10, 56136)
  new_prediction_lengths = (
      lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))
  #new_prediction_lengths shape=(?, 10, 56136)
  # Calculate the scores for each beam
  scores = _get_scores(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight)
  scores_mask = tf.constant([step_log_probs.dtype.min,0],dtype=dtypes.float32,shape=[vocab_size],name='mask')
  scores_masked =tf.add(scores,scores_mask)
  scores_mask2 = tf.constant([0,0,0,0,0,step_log_probs.dtype.min,0],dtype=dtypes.float32,shape=[vocab_size],name='mask2')
  scores_masked =tf.add(scores_mask2,scores_masked)
  def new_scores(scores_masked):
      scores_no_stop = tf.constant([0,0,step_log_probs.dtype.min,0],dtype=dtypes.float32,shape=[vocab_size],name='no_stop')
      scores=tf.add(scores_masked,scores_no_stop)
      return scores
  #constrain the length
  scores = control_flow_ops.cond(
      #time <9 ,
      time <0,
      lambda: new_scores(scores_masked),
      lambda: scores_masked)

  #scores shape=(?, 10, 56136)
  #[batch_size, beam_width, vocab_size]
  time = ops.convert_to_tensor(time, name="time")
  # During the first time step we only consider the initial beam
  scores_shape = array_ops.shape(scores)
  #scores_shape" shape=(3,)
  scores_to_flat_1 =array_ops.reshape(scores, [batch_size,2, -1])
  print ("scores_to_flat_1",scores_to_flat_1)
  scores_to_0 = scores[:, 0]
  scores_to_1 = scores[:, -1]
  scores_to_flat_2=tf.concat([scores_to_0,scores_to_1],1)
  scores_flat = control_flow_ops.cond(
      time > 0,
      lambda: scores_to_flat_1,
      lambda: array_ops.reshape(scores_to_flat_2, [batch_size,2, -1]))
  num_available_beam = control_flow_ops.cond(
      time > 0, lambda: math_ops.reduce_prod(scores_shape[1:]),
      lambda: math_ops.reduce_prod(scores_shape[2:]))
  #scores_flat", shape=(?, ?)
  #num_available_beam" shape=()
  # Pick the next beams according to the specified successors function
  next_beam_size = math_ops.minimum(
      ops.convert_to_tensor(beam_width, dtype=dtypes.int32, name="beam_width"),
      num_available_beam)
  #scores_t = tf.reshape(scores_flat,[batch_size,2,-1])
  ############################
  #input_words=['entrencheds01', 'entrencheds02', 'forgev01', 'forgev04', \
  #             'hitn02', 'hitn03', 'vaultn02', 'vaultn04', 'deepa03', \
  #             'deeps02', 'admitv01', 'admitv02', 'plantn01', 'plantn02',\
  #             'squaren01', 'squaren05', 'drawv05', 'drawv06', 'spellv03', \
  #             'spellv02', 'shotn02', 'shotn04', 'coachv01', 'coachv02', 'casen05',\
  #             'casen09', 'focusn01', 'focusn02', 'tasten01', 'tasten04', 'footn01', \
  #             'footv01']
  input_words=get_words()
  return_list=prior_scores(input_words)
  return_array=np.array(return_list)
  return_tensor=tf.convert_to_tensor(return_array)
  tiling = [1, 5, 1]
  prior_mask=tf.tile(tf.expand_dims(return_tensor, 1), tiling)
  prior_mask=tf.cast(prior_mask, tf.float32)
  prior_mask=array_ops.reshape(prior_mask, [batch_size, -1])
  #print ("prior_mask",prior_mask)
  scores_sum= tf.reduce_sum(scores_to_flat_1,1)
  #print ("scores_sum_1",scores_sum)
  #def cal_scores_sum(scores_sum,prior_mask):
  #    return tf.add(scores_sum,prior_mask)
  #scores_sum = control_flow_ops.cond(
  #    time > 0,
  #    lambda: cal_scores_sum(scores_sum,prior_mask),
  #    lambda: scores_sum)
  #scores_sum=tf.add(scores_sum,prior_mask)
  #print ("scores_sum_2",scores_sum)
  ############################

  #scores_final=tf.concat([scores_sum, scores_sum],1)
  def cal_scores_indices(scores_to_0,scores_to_1):
    next_beam_scores_1, word_indices_1 = nn_ops.top_k(scores_to_0, k=5)
    print ("ori next_beam_scores_1,word_indices_1",next_beam_scores_1)
    print ("ori word_indices_1",word_indices_1)
    next_beam_scores_2, word_indices_2 = nn_ops.top_k(scores_to_1, k=5)
    next_beam_scores=tf.concat([next_beam_scores_1,next_beam_scores_2],1)
    word_indices=tf.concat([word_indices_1,word_indices_2+9*vocab_size],1)
    return next_beam_scores,word_indices
  def cal_scores_indices_t1(scores_final,next_beam_size):
      next_beam_scores_1, word_indices_1=nn_ops.top_k(scores_final, k=5)
      #next_beam_scores_1, word_indices_1=sample(next_beam_scores_1,word_indices_1)
      print ("next_beam_scores_1", next_beam_scores_1)
      print ("word_indices_1",word_indices_1)
      next_beam_scores=tf.concat([next_beam_scores_1,next_beam_scores_1],1)
      word_indices=tf.concat([word_indices_1,word_indices_1+5*vocab_size],1)
      return next_beam_scores, word_indices
  next_beam_scores, word_indices=control_flow_ops.cond(
          time > 0, lambda: cal_scores_indices_t1(scores_sum,next_beam_size),
      lambda: cal_scores_indices(scores_to_0,scores_to_1))

  next_beam_scores.set_shape([static_batch_size, beam_width])
  word_indices.set_shape([static_batch_size, beam_width])
  #shape=(?, ?)
  # Pick out the probs, beam_ids, and states according to the chosen predictions

  next_beam_probs = _tensor_gather_helper(
      gather_indices=word_indices,
      gather_from=total_probs,
      batch_size=batch_size,
      range_size=beam_width * vocab_size,
      gather_shape=[-1],
      name="next_beam_probs")
  # Note: just doing the following
  #   math_ops.to_int32(word_indices % vocab_size,
  #       name="next_beam_word_ids")
  # would be a lot cleaner but for reasons unclear, that hides the results of
  # the op which prevents capturing it with tfdbg debug ops.
  raw_next_word_ids = math_ops.mod(word_indices, vocab_size,
                                   name="next_beam_word_ids")
  #raw_next_word_ids shape=(?, 10)
  next_word_ids = math_ops.to_int32(raw_next_word_ids)
  next_beam_ids = math_ops.to_int32(word_indices / vocab_size,
                                    name="next_beam_parent_ids")

  # Append new ids to current predictions
  previously_finished = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=previously_finished,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_finished = math_ops.logical_or(previously_finished,
                                      math_ops.equal(next_word_ids, end_token),
                                      name="next_beam_finished")

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged
  # 2. Beams that are now finished (EOS predicted) remain unchanged
  # 3. Beams that are not yet finished have their length increased by 1
  lengths_to_add = math_ops.to_int64(
      math_ops.not_equal(next_word_ids, end_token))
  lengths_to_add = (1 - math_ops.to_int64(next_finished)) * lengths_to_add
  next_prediction_len = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.lengths,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_prediction_len += lengths_to_add

  # Pick out the cell_states according to the next_beam_ids. We use a
  # different gather_shape here because the cell_state tensors, i.e.
  # the tensors that would be gathered from, all have dimension
  # greater than two and we need to preserve those dimensions.
  # pylint: disable=g-long-lambda
  next_cell_state = nest.map_structure(
      lambda gather_from: _maybe_tensor_gather_helper(
          gather_indices=next_beam_ids,
          gather_from=gather_from,
          batch_size=batch_size,
          range_size=beam_width,
          gather_shape=[batch_size * beam_width, -1]),
      next_cell_state)
  # pylint: enable=g-long-lambda

  next_state = BeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished)
  print ('next_beam_probs',next_beam_probs)
  output = BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids)

  return output, next_state


def _get_scores(log_probs, sequence_lengths, length_penalty_weight):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape
      `[batch_size, beam_width, vocab_size]`.
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

  Returns:
    The scores normalized by the length_penalty.
  """
  length_penality_ = _length_penalty(
      sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)
  return log_probs / length_penality_


def _length_penalty(sequence_lengths, penalty_factor):
  """Calculates the length penalty. See https://arxiv.org/abs/1609.08144.

  Args:
    sequence_lengths: The sequence length of all hypotheses, a tensor
      of shape [beam_size, vocab_size].
    penalty_factor: A scalar that weights the length penalty.

  Returns:
    The length penalty factor, a tensor fo shape [beam_size].
  """
  penalty_factor = ops.convert_to_tensor(penalty_factor, name="penalty_factor")
  penalty_factor.set_shape(())  # penalty should be a scalar.
  static_penalty = tensor_util.constant_value(penalty_factor)
  if static_penalty is not None and static_penalty == 0:
    return 1.0
  return math_ops.div((5. + math_ops.to_float(sequence_lengths))
                      **penalty_factor, (5. + 1.)**penalty_factor)


def _mask_probs(probs, eos_token, finished):
  """Masks log probabilities.

  The result is that finished beams allocate all probability mass to eos and
  unfinished beams remain unchanged.

  Args:
    probs: Log probabiltiies of shape `[batch_size, beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to.
    finished: A boolean tensor of shape `[batch_size, beam_width]` that
      specifies which
      elements in the beam are finished already.

  Returns:
    A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished
    beams stay unchanged and finished beams are replaced with a tensor with all
    probability on the EOS token.
  """
  vocab_size = array_ops.shape(probs)[2]
  finished_mask = array_ops.expand_dims(
      math_ops.to_float(1. - math_ops.to_float(finished)), 2)
  # These examples are not finished and we leave them
  non_finished_examples = finished_mask * probs
  print ("eos_token",eos_token)
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row_1 = array_ops.one_hot(
      eos_token,
      vocab_size,
      dtype=probs.dtype,
      on_value=0.,
      off_value=probs.dtype.min)
  finised_row_2 = array_ops.one_hot(
      0,
      vocab_size,
      dtype=probs.dtype,
      on_value=abs(probs.dtype.min),
      off_value=0.)
  #finished_row=tf.add(finished_row_1,finised_row_2)
  print ('probs.dtype',probs.dtype)
  #finished_examples = 0
  finished_examples = (1. - finished_mask) * finished_row_1
  #print ("finished_row",finished_row)
  #print ("finished_examples",finished_examples)
  #finished_row shape=(?,), dtype=float32
  #finished_examples shape=(?, 10, ?), dtype=float32
  #non_finished_examples  shape=(?, 10, 56136)
  #finished_examples + non_finished_examples shape=(?, 10, 56136)
  return finished_examples + non_finished_examples


def _maybe_tensor_gather_helper(gather_indices, gather_from, batch_size,
                                range_size, gather_shape):
  """Maybe applies _tensor_gather_helper.

  This applies _tensor_gather_helper when the gather_from dims is at least as
  big as the length of gather_shape. This is used in conjunction with nest so
  that we don't apply _tensor_gather_helper to inapplicable values like scalars.

  Args:
    gather_indices: The tensor indices that we use to gather.
    gather_from: The tensor that we are gathering from.
    batch_size: The batch size.
    range_size: The number of values in each range. Likely equal to beam_width.
    gather_shape: What we should reshape gather_from to in order to preserve the
      correct values. An example is when gather_from is the attention from an
      AttentionWrapperState with shape [batch_size, beam_width, attention_size].
      There, we want to preserve the attention_size elements, so gather_shape is
      [batch_size * beam_width, -1]. Then, upon reshape, we still have the
      attention_size as desired.

  Returns:
    output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
      or the original tensor if its dimensions are too small.
  """
  _check_maybe(gather_from)
  if gather_from.shape.ndims >= len(gather_shape):
    return _tensor_gather_helper(
        gather_indices=gather_indices,
        gather_from=gather_from,
        batch_size=batch_size,
        range_size=range_size,
        gather_shape=gather_shape)
  else:
    return gather_from


def _tensor_gather_helper(gather_indices, gather_from, batch_size,
                          range_size, gather_shape, name=None):
  """Helper for gathering the right indices from the tensor.

  This works by reshaping gather_from to gather_shape (e.g. [-1]) and then
  gathering from that according to the gather_indices, which are offset by
  the right amounts in order to preserve the batch order.

  Args:
    gather_indices: The tensor indices that we use to gather.
    gather_from: The tensor that we are gathering from.
    batch_size: The input batch size.
    range_size: The number of values in each range. Likely equal to beam_width.
    gather_shape: What we should reshape gather_from to in order to preserve the
      correct values. An example is when gather_from is the attention from an
      AttentionWrapperState with shape [batch_size, beam_width, attention_size].
      There, we want to preserve the attention_size elements, so gather_shape is
      [batch_size * beam_width, -1]. Then, upon reshape, we still have the
      attention_size as desired.
    name: The tensor name for set of operations. By default this is
      'tensor_gather_helper'. The final output is named 'output'.

  Returns:
    output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
  """
  with ops.name_scope(name, "tensor_gather_helper"):
    range_ = array_ops.expand_dims(math_ops.range(batch_size) * range_size, 1)
    gather_indices = array_ops.reshape(gather_indices + range_, [-1])
    output = array_ops.gather(
        array_ops.reshape(gather_from, gather_shape), gather_indices)
    final_shape = array_ops.shape(gather_from)[:1 + len(gather_shape)]
    static_batch_size = tensor_util.constant_value(batch_size)
    final_static_shape = (tensor_shape.TensorShape([static_batch_size])
                          .concatenate(
                              gather_from.shape[1:1 + len(gather_shape)]))
    output = array_ops.reshape(output, final_shape, name="output")
    output.set_shape(final_static_shape)
    return output
def get_words():
    words=[]
    with open(PUNGAN_ROOT_PATH + '/Pun_Generation_Forward/data/sample_130','r')as fr:
        for line in fr:
            words.append(line.decode('utf-8').strip())
    return words
def prior_scores(words):
    vocab_dict={}
    ids_list=[]
    cnt=0
    with open(PUNGAN_ROOT_PATH + '/Pun_Generation_Forward/code/inference/final_vocab','r')as fv:
        for line in fv:
            vocab_dict[line.decode('utf-8').strip()]=cnt
            cnt+=1
    for i in words:
        if i in vocab_dict:
            ids_list.append(vocab_dict[i])
        else:
            ids_list.append(2)

    key_word_scores={}
    #with open('inference/tags_res_withscores','r')as fr:
    with open(PUNGAN_ROOT_PATH + '/Pun_Generation_Forward/code/inference/tags_res_PMI','r')as fr:
        for line in fr:
            try:
                word_scores={}
                word_l=line.decode('utf-8').strip().split('\t')[1].split(' ')
                scores_l=line.decode('utf-8').strip().split('\t')[2].split(' ')
            except:
                pass

            for i in range(len(word_l)):
               #idf
               # word_scores[word_l[i]]=10 / (1.0 + np.exp(-float(scores_l[i])/float(scores_l[0])))
                word_scores[word_l[i]]=6 / (1.0 + np.exp(-float(scores_l[i])/float(scores_l[0])))
            if word_scores[word_l[i]] and line.strip().split('\t')[0] in vocab_dict.keys():
                key_word_scores[vocab_dict[line.strip().split('\t')[0]]]=word_scores


    key_id_score={}
    for k1,v1 in key_word_scores.items():
        id_score={}
        for a, b in v1.items():
            try:
                id_score[vocab_dict[a]]=b
            except:
                pass
        key_id_score[k1]=id_score

    key_array={}
    for k,v in key_id_score.items():
        a=np.zeros(105000)
        for k2,v2 in v.items():
            a[k2]=v2
        key_array[k]=a
    return_list=[]
    for i in ids_list:
        if i in key_array.keys():
            return_list.append(key_array[i])
        else:
            return_list.append(np.zeros(105000))
    concat_list=[]
    for i in range(len(ids_list)):
        if i%2 ==0:
            concat_list.append(np.add(return_list[i],return_list[i+1]))
    concat_list.extend(concat_list)
    #print (len(concat_list))
    return concat_list
def sample(energy,word_index, temperature=1.0):
        """sample at most n different elements according to their energy"""
        t_len=energy.shape[0]
        print ("t_len",t_len)
        scores=[]
        word_f_index=[]
        n=5
        n=min(n,32)
        #prb = tf.exp(-np.array(energy)/temperature)
        #prb_mask = tf.constant([prb.dtype.min,0],dtype=dtypes.float32,shape=prb[0].shape,name='mask')
        #prb=tf.add(prb,prb_mask)
        prb = tf.exp(-np.array(energy)/temperature)
        z = tf.reduce_sum(prb,1)
        z=array_ops.reshape(z, [32, -1])
        r=tf.multinomial(prb/z,5)
        for idx in xrange(32):
            scores.append(tf.gather(energy[idx],r[idx]))
            word_f_index.append(tf.gather(word_index[idx],r[idx]))
        scores=tf.convert_to_tensor(scores)
        word_f_index=tf.convert_to_tensor(word_f_index)
        word_f_index=tf.cast(word_f_index,dtype=tf.int32)
        return scores,word_f_index
