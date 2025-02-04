# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.extend
import jax.numpy as jnp

# Enable float8 types
try:
	a = jnp.array(1.0, dtype=jnp.float8_e5m2)  # Test FP8 dtype
	b = jnp.array(2.0, dtype=jnp.float8_e5m2)
	c = a + b  # Perform an operation
	c.block_until_ready()  # Force execution on device
	HAS_FLOAT8 = True
except (RuntimeError, TypeError):
	HAS_FLOAT8 = False

DTYPE_MAPPING = {
	# Standard types
	"bf16": jnp.bfloat16,
	"f16": jnp.float16,
	"f32": jnp.float32,
	"f64": jnp.float64,
	"bfloat16": jnp.bfloat16,
	"float16": jnp.float16,
	"float32": jnp.float32,
	"float64": jnp.float64,
}

# Add float8 types if available
if HAS_FLOAT8:
	DTYPE_MAPPING.update(
		{
			"f8_e4m3": jnp.float8_e4m3fn,
			"f8_e5m2": jnp.float8_e5m2,
			"float8_e4m3": jnp.float8_e4m3fn,
			"float8_e5m2": jnp.float8_e5m2,
		}
	)


def get_platform_default_half() -> jnp.dtype:
	"""Returns platform-specific half precision type."""
	platform = jax.extend.backend.get_backend().platform
	return jnp.bfloat16 if platform == "tpu" else jnp.float16


DTYPE_MAPPING["half"] = get_platform_default_half()
