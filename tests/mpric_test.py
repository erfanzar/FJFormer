import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
import jax.numpy as jnp
import pytest

from eformer.mpric import (
	DTYPE_MAPPING,
	HAS_FLOAT8,
	LossScaleConfig,
	Policy,
	PrecisionHandler,
)


@pytest.fixture
def sample_data():
	"""Provide sample data for testing."""
	key = jax.random.PRNGKey(0)
	x = jax.random.normal(key, (10, 5))
	y = jax.random.normal(key, (10, 1))
	return x, y


@pytest.fixture
def simple_model():
	"""Provide a simple linear model for testing."""

	def model(params, x):
		return jnp.dot(x, params["w"]) + params["b"]

	return model


@pytest.fixture
def model_params():
	"""Initialize model parameters."""
	return {"w": jnp.ones((5, 1)), "b": jnp.zeros(1)}


def test_policy_creation():
	"""Test policy creation from different formats."""
	# Test string format
	policy = Policy.from_string("p=f32,c=bf16,o=f32")
	assert policy.param_dtype == jnp.float32
	assert policy.compute_dtype == jnp.bfloat16
	assert policy.output_dtype == jnp.float32

	# Test single dtype format
	policy = Policy.from_string("f32")
	assert policy.param_dtype == jnp.float32
	assert policy.compute_dtype == jnp.float32
	assert policy.output_dtype == jnp.float32

	# Test invalid dtype
	with pytest.raises(ValueError):
		Policy.from_string("p=invalid_dtype")


@pytest.mark.skipif(not HAS_FLOAT8, reason="float8 not available")
def test_float8_support():
	"""Test float8 dtype support."""
	policy = Policy.from_string("p=f32,c=f8_e4m3,o=f32")
	assert policy.compute_dtype == jnp.float8_e4m3fn

	policy = Policy.from_string("p=f32,c=f8_e5m2,o=f32")
	assert policy.compute_dtype == jnp.float8_e5m2


def test_precision_casting(sample_data):
	"""Test precision casting operations."""
	x, _ = sample_data
	handler = PrecisionHandler("p=f32,c=bf16,o=f32")

	# Test compute precision casting
	x_compute = handler.cast_for_compute(x)
	assert x_compute.dtype == jnp.bfloat16

	# Test output precision casting
	x_output = handler.cast_for_output(x_compute)
	assert x_output.dtype == jnp.float32

	# Test parameter casting
	params = {"w": jnp.ones((5, 1), dtype=jnp.float32)}
	params_cast = handler.cast_params(params)
	assert params_cast["w"].dtype == jnp.float32


def test_loss_scaling(sample_data, simple_model, model_params):
	"""Test loss scaling during training."""
	x, y = sample_data

	# Initialize handler with dynamic loss scaling
	handler = PrecisionHandler(
		"p=f32,c=bf16,o=f32",
		use_dynamic_scale=True,
		loss_scale_config=LossScaleConfig(
			initial_scale=2.0, growth_interval=2, scale_factor=2
		),
	)

	# Define training step
	def training_step(params, x, y):
		def loss_fn(params):
			y_pred = simple_model(params, x)
			return jnp.mean((y_pred - y) ** 2)

		loss = loss_fn(params)
		grads = jax.grad(loss_fn)(params)
		return loss, grads

	# Wrap training step
	wrapped_step = handler.training_step_wrapper(training_step)

	# Run multiple steps to test loss scaling behavior
	for _ in range(5):
		loss, grads, grads_finite = wrapped_step(model_params, x, y)

		# Check types
		assert loss.dtype == jnp.float32
		assert all(g.dtype == jnp.float32 for g in jax.tree_util.tree_leaves(grads))
		assert isinstance(grads_finite, jnp.ndarray)
		assert grads_finite.dtype == jnp.bool_


def test_inference_wrapper(sample_data, simple_model, model_params):
	"""Test inference wrapper."""
	x, _ = sample_data

	handler = PrecisionHandler("p=f32,c=bf16,o=f32")

	def inference_fn(params, x):
		return simple_model(params, x)

	wrapped_inference = handler.inference_wrapper(inference_fn)

	output = wrapped_inference(model_params, x)
	assert output.dtype == jnp.float32


def test_dynamic_loss_scaling_adjustment():
	"""Test dynamic loss scaling behavior."""
	handler = PrecisionHandler(
		"f32",
		use_dynamic_scale=True,
		loss_scale_config=LossScaleConfig(
			initial_scale=2.0, growth_interval=2, scale_factor=2
		),
	)

	for _ in range(2):
		_, _, grads_finite = handler.training_step_wrapper(
			lambda: (jnp.array(1.0), {"w": jnp.array(1.0)})
		)()
		assert grads_finite

	assert handler.loss_scaler.loss_scale > 2.0

	def training_step_with_nan():
		return jnp.array(1.0), {"w": jnp.array(jnp.nan)}

	_, _, grads_finite = handler.training_step_wrapper(training_step_with_nan)()
	assert not grads_finite

	assert handler.loss_scaler.loss_scale < 4.0


def test_error_handling():
	"""Test error handling in the precision handler."""
	with pytest.raises(ValueError):
		PrecisionHandler("invalid_policy")

	with pytest.raises(ValueError):
		PrecisionHandler("p=f16,c=invalid,o=f32")



# File: tests/test_dtypes.py
def test_dtype_mapping():
	"""Test dtype mapping functionality."""

	# Test basic dtypes
	assert DTYPE_MAPPING["f32"] == jnp.float32
	assert DTYPE_MAPPING["bf16"] == jnp.bfloat16

	# Test platform-specific half precision
	half_dtype = DTYPE_MAPPING["half"]
	assert half_dtype in (jnp.float16, jnp.bfloat16)


if __name__ == "__main__":
	pytest.main([__file__])
