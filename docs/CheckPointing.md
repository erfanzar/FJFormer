* **StreamingCheckpointer** class:
    * **__init__(config, checkpoint_dir, enable=True)**:
      Initializes a StreamingCheckpointer object.
      Args:
      config: A dictionary of configuration options.
      checkpoint_dir: The directory where checkpoints will be saved.
      enable: Whether to enable the streaming checkpointing functionality.
    * **save_checkpoint(train_state, filename, gather_fns=None)**:
      Saves a checkpoint to the specified file.
      Args:
      train_state: The train state to save.
      filename: The name of the checkpoint file.
      gather_fns: A dictionary of functions that can be used to gather
      large tensors into smaller chunks before saving them.
    * **save_all(train_state, gather_fns, metadata=None, dataset=None, milestone=False)**:
      Saves a checkpoint for the current step, as well as metadata and dataset
      information.
      Args:
      train_state: The train state to save.
      gather_fns: A dictionary of functions that can be used to gather
      large tensors into smaller chunks before saving them.
      metadata: Metadata to save.
      dataset: Dataset information to save.
      milestone: Whether this is a milestone checkpoint.
    * **load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None)**:
      Loads a checkpoint from the specified file.
      Args:
      path: The path to the checkpoint file.
      target: The object to load the checkpoint into.
      shard_fns: A dictionary of functions that can be used to shard
      tensors after loading them.
      remove_dict_prefix: A tuple of keys to remove from the loaded
      checkpoints.
    * **load_flax_checkpoint(path, target=None, shard_fns=None)**:
      Loads a standard flax checkpoint from the specified file.
      Args:
      path: The path to the checkpoint file.
      target: The object to load the checkpoint into.
      shard_fns: A dictionary of functions that can be used to shard
      tensors after loading them.

* **load_trainstate_checkpoint(load_from, trainstate_target=None,
  trainstate_shard_fns=None,
  disallow_trainstate=False)**:
  Load a train state checkpoint from the specified load_from string.
  Args:
  load_from: The load_from string, which can be one of the following:
  * 'trainstate': Load the entire train state.
  * 'trainstate_params': Load the params part of the train state.
  * 'params': Load the params.
  * 'flax_params': Load the params in the standard flax format (non-streaming).
  trainstate_target: The target object to load the train state into.
  trainstate_shard_fns: A dictionary of functions that can be used to shard
  tensors after loading them.
  disallow_trainstate: Whether to disallow loading the full train state.
