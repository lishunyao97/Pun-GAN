# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""For training NMT models."""
from __future__ import print_function

import math
import os
import random
import time

import tensorflow as tf

import attention_model
import gnmt_model
import inference
import model as nmt_model
import model_helper
from utils import misc_utils as utils
from utils import nmt_utils
from nltk.corpus import wordnet as wn
utils.check_tensorflow_version()
PUNGAN_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__all__ = [
    "run_sample_decode", "run_internal_eval", "run_external_eval",
    "run_avg_external_eval", "run_full_eval", "init_stats", "update_stats",
    "print_step_info", "process_stats", "train"
]


def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data, tgt_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder, summary_writer)

def run_sample_decode_pungan(infer_model, infer_sess, model_dir, hparams, src_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  return _sample_decode_pungan(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder)

def run_internal_eval(
    eval_model, eval_sess, model_dir, hparams, summary_writer,
    use_test_set=True):
  """Compute internal evaluation (perplexity) for both dev / test."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_eval_iterator_feed_dict = {
      eval_model.src_file_placeholder: dev_src_file,
      eval_model.tgt_file_placeholder: dev_tgt_file
  }

  dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                           eval_model.iterator, dev_eval_iterator_feed_dict,
                           summary_writer, "dev")
  test_ppl = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: test_src_file,
        eval_model.tgt_file_placeholder: test_tgt_file
    }
    test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                              eval_model.iterator, test_eval_iterator_feed_dict,
                              summary_writer, "test")
  with open('forward_ppl.txt', 'a+') as fw:
      fw.write('dev_ppl: ')
      fw.write(str(round(dev_ppl, 2)))
      fw.write('\ttest_ppl: ')
      fw.write(str(round(test_ppl, 2))+'\n')

  return dev_ppl, test_ppl


def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True, use_test_set=True,
                      avg_ckpts=False):
  """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_infer_iterator_feed_dict = {
      infer_model.src_placeholder: inference.load_data(dev_src_file),
      infer_model.batch_size_placeholder: hparams.infer_batch_size,
  }
  dev_scores = _external_eval(
      loaded_infer_model,
      global_step,
      infer_sess,
      hparams,
      infer_model.iterator,
      dev_infer_iterator_feed_dict,
      dev_tgt_file,
      "dev",
      summary_writer,
      save_on_best=save_best_dev,
      avg_ckpts=avg_ckpts)

  test_scores = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_infer_iterator_feed_dict = {
        infer_model.src_placeholder: inference.load_data(test_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    test_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        test_infer_iterator_feed_dict,
        test_tgt_file,
        "test",
        summary_writer,
        save_on_best=False,
        avg_ckpts=avg_ckpts)
  return dev_scores, test_scores, global_step


def run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                          summary_writer, global_step):
  """Creates an averaged checkpoint and run external eval with it."""
  avg_dev_scores, avg_test_scores = None, None
  if hparams.avg_ckpts:
    # Convert VariableName:0 to VariableName.
    global_step_name = infer_model.model.global_step.name.split(":")[0]
    avg_model_dir = model_helper.avg_checkpoints(
        model_dir, hparams.num_keep_ckpts, global_step, global_step_name)

    if avg_model_dir:
      avg_dev_scores, avg_test_scores, _ = run_external_eval(
          infer_model,
          infer_sess,
          avg_model_dir,
          hparams,
          summary_writer,
          avg_ckpts=True)

  return avg_dev_scores, avg_test_scores


def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data,
                  avg_ckpts=False):
  """Wrapper for running sample_decode, internal_eval and external_eval."""
  run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                    sample_src_data, sample_tgt_data)
  dev_ppl, test_ppl = run_internal_eval(
      eval_model, eval_sess, model_dir, hparams, summary_writer)
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model, infer_sess, model_dir, hparams, summary_writer)

  metrics = {
      "dev_ppl": dev_ppl,
      "test_ppl": test_ppl,
      "dev_scores": dev_scores,
      "test_scores": test_scores,
  }

  avg_dev_scores, avg_test_scores = None, None
  if avg_ckpts:
    avg_dev_scores, avg_test_scores = run_avg_external_eval(
        infer_model, infer_sess, model_dir, hparams, summary_writer,
        global_step)
    metrics["avg_dev_scores"] = avg_dev_scores
    metrics["avg_test_scores"] = avg_test_scores

  result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
  if avg_dev_scores:
    result_summary += ", " + _format_results("avg_dev", None, avg_dev_scores,
                                             hparams.metrics)
  if hparams.test_prefix:
    result_summary += ", " + _format_results("test", test_ppl, test_scores,
                                             hparams.metrics)
    if avg_test_scores:
      result_summary += ", " + _format_results("avg_test", None,
                                               avg_test_scores, hparams.metrics)

  return result_summary, global_step, metrics


def init_stats():
  """Initialize statistics that we want to accumulate."""
  return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
          "total_count": 0.0, "grad_norm": 0.0}


def update_stats(stats, start_time, step_result, hparams):
  """Update stats: write summary and accumulate statistics."""
  (_, step_loss, step_predict_count, step_summary, global_step,
   step_word_count, batch_size, grad_norm, learning_rate) = step_result

  # Update statistics
  stats["step_time"] += (time.time() - start_time)
  stats["loss"] += (step_loss * batch_size)
  stats["predict_count"] += step_predict_count
  stats["total_count"] += float(step_word_count)
  stats["grad_norm"] += grad_norm
  return global_step, learning_rate, step_summary


def print_step_info(prefix, global_step, info, result_summary, log_f):
  """Print all info at the current global step."""
  utils.print_out(
      "%sstep %d lr %g step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["speed"], info["train_ppl"], info["avg_grad_norm"], result_summary,
       time.ctime()),
      log_f)


def process_stats(stats, info, global_step, steps_per_stats, log_f):
  """Update info and check for overflow."""
  # Update info
  info["avg_step_time"] = stats["step_time"] / steps_per_stats
  info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
  info["train_ppl"] = utils.safe_exp(stats["loss"] / stats["predict_count"])
  info["speed"] = stats["total_count"] / (1000 * stats["step_time"])

  # Check for overflow
  is_overflow = False
  train_ppl = info["train_ppl"]
  if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
    utils.print_out("  step %d overflow, stop early" % global_step,
                    log_f)
    is_overflow = True

  return is_overflow


def before_train(loaded_train_model, train_model, train_sess, global_step,
                 hparams, log_f):
  """Misc tasks to do before training."""
  stats = init_stats()
  info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
          "avg_grad_norm": 0.0,
          "learning_rate": loaded_train_model.learning_rate.eval(
              session=train_sess)}
  start_train_time = time.time()
  utils.print_out("# Start step %d, lr %g, %s" %
                  (global_step, info["learning_rate"], time.ctime()), log_f)

  # Initialize all of the iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  train_sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: skip_count})

  return stats, info, start_train_time

def run_sample_decode_pungan_prepare(hparams, scope=None, target_session=""):
    log_device_placement = hparams.log_device_placement
    out_dir = hparams.out_dir
    if not hparams.attention: # choose this model
        model_creator = nmt_model.Model
    else:  # Attention
        if (hparams.encoder_type == "gnmt" or hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
            model_creator = gnmt_model.GNMTModel
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        else:
            raise ValueError("Unknown attention architecture %s" % hparams.attention_architecture)
    infer_model = model_helper.create_infer_model(model_creator, hparams, scope)
    def dealt(input, output):
        with open(input) as f:
            with open(output,'w') as fw:
                for line in f:
                    l = line.strip().split()
                    l.reverse()
                    sent = ' '.join(l)
                    fw.write(sent+'\n')
    wsd_src_file = "%s" % (hparams.sample_prefix)
    wsd_src_file_new = wsd_src_file+'.new'
    dealt(wsd_src_file,wsd_src_file_new)

    wsd_src_data = inference.load_data(wsd_src_file_new)
    model_dir = hparams.out_dir
    # TensorFlow model
    config_proto = utils.get_config_proto(
        log_device_placement=log_device_placement,
        num_intra_threads=hparams.num_intra_threads,
        num_inter_threads=hparams.num_inter_threads)
    infer_sess = tf.Session(
        target=target_session, config=config_proto, graph=infer_model.graph)
    print('len wsd_src_data', len(wsd_src_data))
    eval_result = []
    for i in range(len(wsd_src_data)/32):
        eval_result += run_sample_decode_pungan(infer_model, infer_sess, model_dir, hparams, wsd_src_data[i*32:(i+1)*32])
    print('eval_result')
    print(eval_result)
    print(len(eval_result))
    backward_step1_in = []
    with open(PUNGAN_ROOT_PATH + '/Pun_Generation/data/1backward/backward_step1.in') as f:
        for line in f:
            backward_step1_in.append(line.strip())
    def wsd_input_format(wsd_src_data, eval_result):
        '''
        test_data[0] {'target_word': u'art#n', 'target_sense': None, 'id': 'senseval2.d000.s000.t000', 'context': ['the', '<target>', 'of', 'change_ringing', 'be', 'peculiar', 'to', 'the', 'english', ',', 'and', ',', 'like', 'most', 'english', 'peculiarity', ',', 'unintelligible', 'to', 'the', 'rest', 'of', 'the', 'world', '.'], 'poss': ['DET', 'NOUN', 'ADP', 'NOUN', 'VERB', 'ADJ', 'PRT', 'DET', 'NOUN', '.', 'CONJ', '.', 'ADP', 'ADJ', 'ADJ', 'NOUN', '.', 'ADJ', 'PRT', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', '.']}
        '''
        wsd_input = []
        senses_input = []

        for i in range(len(eval_result)):
            block = i/32
            src_word1, src_word2 = backward_step1_in[2*block], backward_step1_in[2*block+1]
            tgt_sent = wsd_src_data[i].decode().encode('utf-8') + ' ' + eval_result[i]
            tgt_word = src_word1


            synset = wn.lemma_from_key(tgt_word).synset()
            s = synset.name()
            target_word = '#'.join(s.split('.')[:2])
            context = tgt_sent.split(' ')

            for j in range(len(context)):
                if context[j] == tgt_word:
                    context[j] = '<target>'
            poss_list = ['.' for _ in range(len(context))]
            tmp_dict = {'target_word': target_word, 'target_sense': None, 'id': None, 'context': context, 'poss': poss_list}
            wsd_input.append(tmp_dict)
            senses_input.append((src_word1, src_word2))
        return wsd_input, senses_input
    wsd_input, senses_input = wsd_input_format(wsd_src_data, eval_result)
    print('wsd_input', wsd_input)
    print("len of wsd_input", len(wsd_input))
    return wsd_input, senses_input, wsd_src_data, eval_result


def train(hparams, scope=None, target_session="", compute_ppl=0):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  avg_ckpts = hparams.avg_ckpts

  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  if not hparams.attention: # choose this model
    model_creator = nmt_model.Model
  else:  # Attention
    if (hparams.encoder_type == "gnmt" or
        hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
      model_creator = gnmt_model.GNMTModel
    elif hparams.attention_architecture == "standard":
      model_creator = attention_model.AttentionModel
    else:
      raise ValueError("Unknown attention architecture %s" %
                       hparams.attention_architecture)

  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  # Preload data for sample decoding.
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  sample_src_data = inference.load_data(dev_src_file)
  sample_tgt_data = inference.load_data(dev_tgt_file)
  wsd_src_file = "%s" % (hparams.sample_prefix)

  wsd_src_data = inference.load_data(wsd_src_file)

  summary_name = "train_log"
  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file=%s" % log_file, log_f)

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement,
      num_intra_threads=hparams.num_intra_threads,
      num_inter_threads=hparams.num_inter_threads)
  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(
      target=target_session, config=config_proto, graph=infer_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  # First evaluation
  '''
  run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data, avg_ckpts)
  '''
  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step

  # This is the training loop.
  stats, info, start_train_time = before_train(
      loaded_train_model, train_model, train_sess, global_step, hparams, log_f)
  end_step = global_step+100
  while global_step < end_step: # num_train_steps
    ### Run a step ###
    start_time = time.time()
    try:
      # then forward inference result to WSD, get reward
      step_result = loaded_train_model.train(train_sess)
      # forward reward to placeholder of loaded_train_model, and write a new train function where loss = loss*reward
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      utils.print_out(
          "# Finished an epoch, step %d. Perform external evaluation" %
          global_step)

      # run_sample_decode(infer_model, infer_sess, model_dir, hparams,
      #                   summary_writer, sample_src_data, sample_tgt_data)

      # only for pretrain
      # run_external_eval(infer_model, infer_sess, model_dir, hparams,
      #                   summary_writer)

      if avg_ckpts:
        run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                              summary_writer, global_step)

      train_sess.run(
          train_model.iterator.initializer,
          feed_dict={train_model.skip_count_placeholder: 0})

      continue

    # Process step_result, accumulate stats, and write summary
    global_step, info["learning_rate"], step_summary = update_stats(
        stats, start_time, step_result, hparams)
    summary_writer.add_summary(step_summary, global_step)
    if compute_ppl:
        run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer)
    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
      last_stats_step = global_step
      is_overflow = process_stats(
          stats, info, global_step, steps_per_stats, log_f)
      print_step_info("  ", global_step, info, _get_best_results(hparams),
                      log_f)
      if is_overflow:
        break

      # Reset statistics
      stats = init_stats()

    if global_step - last_eval_step >= steps_per_eval:
      last_eval_step = global_step
      utils.print_out("# Save eval, global step %d" % global_step)
      utils.add_summary(summary_writer, global_step, "train_ppl",
                        info["train_ppl"])

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

      # Evaluate on dev/test
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      run_internal_eval(
          eval_model, eval_sess, model_dir, hparams, summary_writer)

    if global_step - last_external_eval_step >= steps_per_external_eval:
      last_external_eval_step = global_step

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      run_external_eval(
          infer_model, infer_sess, model_dir,
          hparams, summary_writer)

      if avg_ckpts:
        run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                              summary_writer, global_step)

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "translate.ckpt"),
      global_step=global_step)
  '''
  (result_summary, _, final_eval_metrics) = (
      run_full_eval(
          model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
          summary_writer, sample_src_data, sample_tgt_data, avg_ckpts))
  print_step_info("# Final, ", global_step, info, result_summary, log_f)
  utils.print_time("# Done training!", start_train_time)

  summary_writer.close()

  utils.print_out("# Start evaluating saved best models.")
  for metric in hparams.metrics:
    best_model_dir = getattr(hparams, "best_" + metric + "_dir")
    summary_writer = tf.summary.FileWriter(
        os.path.join(best_model_dir, summary_name), infer_model.graph)
    result_summary, best_global_step, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data)
    print_step_info("# Best %s, " % metric, best_global_step, info,
                    result_summary, log_f)
    summary_writer.close()

    if avg_ckpts:
      best_model_dir = getattr(hparams, "avg_best_" + metric + "_dir")
      summary_writer = tf.summary.FileWriter(
          os.path.join(best_model_dir, summary_name), infer_model.graph)
      result_summary, best_global_step, _ = run_full_eval(
          best_model_dir, infer_model, infer_sess, eval_model, eval_sess,
          hparams, summary_writer, sample_src_data, sample_tgt_data)
      print_step_info("# Averaged Best %s, " % metric, best_global_step, info,
                      result_summary, log_f)
      summary_writer.close()

  return final_eval_metrics, global_step
  '''

def _format_results(name, ppl, scores, metrics):
  """Format results."""
  result_str = ""
  if ppl:
    result_str = "%s ppl %.2f" % (name, ppl)
  if scores:
    for metric in metrics:
      if result_str:
        result_str += ", %s %s %.1f" % (name, metric, scores[metric])
      else:
        result_str = "%s %s %.1f" % (name, metric, scores[metric])
  return result_str


def _get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  return ppl


def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
  """Pick a sentence and decode."""
  decode_id = random.randint(0, len(src_data) - 1)
  utils.print_out("  # %d" % decode_id)

  iterator_feed_dict = {
      iterator_src_placeholder: [src_data[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  infer_logits, nmt_outputs, attention_summary = model.decode(sess)
  if hparams.beam_width > 0:
    # get the top translation.
    nmt_outputs = nmt_outputs[0]

  translation = nmt_utils.get_translation(
      nmt_outputs,
      infer_logits,
      sent_id=0,
      tgt_eos=hparams.eos,
      subword_option=hparams.subword_option)

  utils.print_out("    src: %s" % src_data[decode_id])
  utils.print_out("    ref: %s" % tgt_data[decode_id])
  utils.print_out(b"    nmt: " + translation)

  # Summary
  if attention_summary is not None:
    summary_writer.add_summary(attention_summary, global_step)

def _sample_decode_pungan(model, global_step, sess, hparams, iterator, src_data, iterator_src_placeholder, iterator_batch_size_placeholder):
  """Pick a sentence and decode."""
  iterator_feed_dict = {
      iterator_src_placeholder: src_data,
      iterator_batch_size_placeholder: len(src_data),
  }

  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  infer_logits, nmt_outputs, attention_summary = model.decode(sess)
  print('nmt_outputs len', len(nmt_outputs))
  print(nmt_outputs[0])
  if hparams.beam_width > 0:
    # get the top translation.
    nmt_outputs = nmt_outputs[0]
  return_list = []
  for sent_id in range(len(src_data)):
      tgt_eos = hparams.eos
      if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
      output = nmt_outputs[sent_id, :].tolist()
      if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]
      return_list.append(' '.join(output))

  return return_list

def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best, avg_ckpts=False):
  """External evaluation such as BLEU and ROUGE scores."""
  out_dir = hparams.out_dir
  decode = global_step > 0

  if avg_ckpts:
    label = "avg_" + label

  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)

  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  output = os.path.join(out_dir, "output_%s" % label)
  scores = nmt_utils.decode_and_evaluate(
      label,
      model,
      sess,
      output,
      ref_file=tgt_file,
      metrics=hparams.metrics,
      subword_option=hparams.subword_option,
      beam_width=hparams.beam_width,
      tgt_eos=hparams.eos,
      decode=decode)
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      if avg_ckpts:
        best_metric_label = "avg_best_" + metric
      else:
        best_metric_label = "best_" + metric

      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # metric: larger is better
      if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
        setattr(hparams, best_metric_label, scores[metric])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, best_metric_label + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores
