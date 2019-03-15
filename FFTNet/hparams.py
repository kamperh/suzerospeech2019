from collections import namedtuple

Hparams = namedtuple('Hparams', ['num_mels', 'num_freq', 'mcep_dim', 'mcep_alpha',
                                 'minf0', 'maxf0', 'sample_rate', 'feature_type',
                                 'frame_length_ms', 'frame_shift_ms', 'preemphasis',
                                 'min_level_db', 'ref_level_db', 'noise_injecting',
                                 'use_cuda', 'use_local_condition', 'batch_size', 'sample_size',
                                 'learning_rate', 'training_steps', 'checkpoint_interval',
                                 'n_stacks', 'fft_channels', 'quantization_channels'])


# Default hyperparameters:
hparams = Hparams(
  # Audio:
  num_mels=45,
  num_freq=1025,
  mcep_dim=24,
  mcep_alpha=0.41,
  minf0=40,
  maxf0=500,
  sample_rate=16000,
  feature_type='melspc',  # mcc or melspc
  frame_length_ms=25,
  frame_shift_ms=10,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  noise_injecting=True,

  # Training:
  use_cuda=False,
  use_local_condition=True,
  batch_size=6,
  sample_size=16000,
  learning_rate=2e-4,
  training_steps=200000,
  checkpoint_interval=5000,

  # Model
  n_stacks=11,
  fft_channels=256,
  quantization_channels=256,
)
