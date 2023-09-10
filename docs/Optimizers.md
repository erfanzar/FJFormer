* **optax_add_scheduled_weight_decay** function:
    * **Arguments:**
        * **schedule_fn:** A function that takes the current step number as input and returns the weight decay value.
        * **mask:** An optional mask that can be used to apply weight decay to a subset of parameters.
    * **Returns:**
      An Optax GradientTransformation object that adds the scheduled weight decay to the updates.

* **get_adamw_with_cosine_scheduler** function:

```python
tx, scheduler = get_adamw_with_cosine_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate:** The initial learning rate.
    * **b1:** The first Adam beta parameter.
    * **b2:** The second Adam beta parameter.
    * **eps:** The Adam epsilon parameter.
    * **eps_root:** The Adam epsilon root parameter.
    * **weight_decay:** The weight decay coefficient.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **mu_dtype:** The dtype of the Adam momentum terms.
* **Returns:**
  A tuple of the Adam optimizer and the cosine learning rate scheduler.

* **get_adamw_with_linear_scheduler** function:

```python
tx, scheduler = get_adamw_with_linear_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate_start:** The initial learning rate.
    * **learning_rate_end:** The final learning rate.
    * **b1:** The first Adam beta parameter.
    * **b2:** The second Adam beta parameter.
    * **eps:** The Adam epsilon parameter.
    * **eps_root:** The Adam epsilon root parameter.
    * **weight_decay:** The weight decay coefficient.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **mu_dtype:** The dtype of the Adam momentum terms.
* **Returns:**
  A tuple of the Adam optimizer and the linear learning rate scheduler.

* **get_adafactor_with_linear_scheduler** function:

```python
tx, scheduler = get_adafactor_with_linear_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate_start:** The initial learning rate.
    * **learning_rate_end:** The final learning rate.
    * **weight_decay:** The weight decay coefficient.
    * **min_dim_size_to_factor:** The minimum size of a parameter tensor for it to be factored by Adafactor.
    * **decay_rate:** The decay rate parameter for Adafactor.
    * **decay_offset:** The decay offset parameter for Adafactor.
    * **multiply_by_parameter_scale:** Whether to multiply the learning rate by the parameter scale.
    * **clipping_threshold:** The gradient clipping threshold.
    * **momentum:** The momentum parameter for Adafactor.
    * **dtype_momentum:** The dtype of the momentum term for Adafactor.
    * **weight_decay_rate:** The weight decay rate for Adafactor.
    * **eps:** The epsilon parameter for Adafactor.
    * **factored:** Whether to use the factored implementation of Adafactor.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **weight_decay_mask:** A mask tensor that specifies which parameters to apply weight decay to.
* **Returns:**
  A tuple of the Adafactor optimizer and the linear learning rate scheduler.

* **get_adafactor_with_cosine_scheduler** function:

```python
tx, scheduler = get_adafactor_with_cosine_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate:** The initial learning rate.
    * **weight_decay:** The weight decay coefficient.
    * **min_dim_size_to_factor:** The minimum size of a parameter tensor for it to be factored by Adafactor.
    * **decay_rate:** The decay rate parameter for Adafactor.
    * **decay_offset:** The decay offset parameter for Adafactor.
    * **multiply_by_parameter_scale:** Whether to multiply the learning rate by the parameter scale.
    * **clipping_threshold:** The gradient clipping threshold.
    * **momentum:** The momentum parameter for Adafactor.
    * **dtype_momentum:** The dtype of the momentum term for Adafactor.
    * **weight_decay_rate:** The weight decay rate for Adafactor.
    * **eps:** The epsilon parameter for Adafactor.
    * **factored:** Whether to use the factored implementation of Adafactor.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **weight_decay_mask:** A mask tensor that specifies which parameters to apply weight decay to.
* **Returns:**
  A tuple of the Adafactor optimizer and the cosine learning rate scheduler.

* **get_lion_with_cosine_scheduler** function:

```python
tx, scheduler = get_lion_with_cosine_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate:** The initial learning rate.
    * **alpha:** The minimum value of the multiplier used to adjust the learning rate.
    * **exponent:** The exponent of the cosine decay schedule.
    * **b1:** The first Lion beta parameter.
    * **b2:** The second Lion beta parameter.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **mu_dtype:** The dtype of the Lion momentum terms.
* **Returns:**
  A tuple of the Lion optimizer and the cosine learning rate scheduler.

* **get_lion_with_linear_scheduler** function:

```python
tx, scheduler = get_lion_with_linear_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate_start:** The initial learning rate.
    * **learning_rate_end:** The final learning rate.
    * **b1:** The first Lion beta parameter.
    * **b2:** The second Lion beta parameter.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **mu_dtype:** The dtype of the Lion momentum terms.
* **Returns:**
  A tuple of the Lion optimizer and the linear learning rate scheduler.

[//]: # (* **get_adamw_with_warm_up_cosine_scheduler** function:)

* **get_lion_with_linear_scheduler** function:

```python
tx, scheduler = get_lion_with_linear_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate:** The initial learning rate.
    * **b1:** The first Adam beta parameter.
    * **b2:** The second Adam beta parameter.
    * **eps:** The Adam epsilon parameter.
    * **eps_root:** The Adam epsilon root parameter.
    * **weight_decay:** The weight decay coefficient.
    * **exponent:** The exponent of the cosine decay schedule.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **mu_dtype:** The dtype of the Adam momentum terms.
* **Returns:**
  A tuple of the Adam optimizer and the cosine learning rate scheduler.

* **get_adafactor_with_warm_up_cosine_scheduler** function:

```python
tx, scheduler = get_adafactor_with_warm_up_cosine_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate:** The initial learning rate.
    * **weight_decay:** The weight decay coefficient.
    * **min_dim_size_to_factor:** The minimum size of a parameter tensor for it to be factored by Adafactor.
    * **decay_rate:** The decay rate parameter for Adafactor.
    * **decay_offset:** The decay offset parameter for Adafactor.
    * **multiply_by_parameter_scale:** Whether to multiply the learning rate by the parameter scale.
    * **clipping_threshold:** The gradient clipping threshold.
    * **momentum:** The momentum parameter for Adafactor.
    * **dtype_momentum:** The dtype of the momentum term for Adafactor.
    * **weight_decay_rate:** The weight decay rate for Adafactor.
    * **eps:** The epsilon parameter for Adafactor.
    * **factored:** Whether to use the factored implementation of Adafactor.
    * **exponent:** The exponent of the cosine decay schedule.
    * **weight_decay_mask:** A mask tensor that specifies which parameters to apply weight decay to.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
* **Returns:**
  A tuple of the Adafactor optimizer and the cosine learning rate scheduler.

* **get_lion_with_warm_up_cosine_scheduler** function:

```python
tx, scheduler = get_lion_with_warm_up_cosine_scheduler(*args, **kwargs)
```

* **Arguments:**
    * **steps:** The total number of training steps.
    * **learning_rate:** The initial learning rate.
    * **exponent:** The exponent of the cosine decay schedule.
    * **b1:** The first Lion beta parameter.
    * **b2:** The second Lion beta parameter.
    * **gradient_accumulation_steps:** The number of gradient accumulation steps.
    * **mu_dtype:** The dtype of the Lion momentum terms.
* **Returns:**
  A tuple of the Lion optimizer and the cosine learning rate scheduler.

The references for these functions are:

* Lion: A Linear-Complexity Adaptive Learning Rate Method: https://arxiv.org/abs/2204.02267
* Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
* Cosine Annealing with Restarts for Stochastic Optimization: https://arxiv.org/abs/1608.03983
* Adafactor: Adaptive Learning Rates for Neural Networks: https://arxiv.org/abs/1804.04235

I hope this documentation is helpful. Let me know if you have any other questions.