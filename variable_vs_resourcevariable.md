
# 变量体系

* Variable
```python
class VariableV1(Variable):

class RefVariable(VariableV1):
  def __init__:
    self._variable = state_ops.variable_op_v2(
              shape,
              self._initial_value.dtype.base_dtype,
              name=name)
              
class 
```
* ResourceVariable
```python

class ResourceVariable(VariableV1):
  def __init__:
    self._handle = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                        shared_name=shared_name,
                        name=name,
                        container=container)
    self._initializer_op = (
                  gen_resource_variable_ops.assign_variable_op(
                      self._handle,
                      variables._try_guard_against_uninitialized_dependencies(
                          name,
                          initial_value),
                      name=n))
  
  def value(self):
    return self._read_variable_op()
    
  def _read_variable_op(self):
    return gen_resource_variable_ops.read_variable_op(self._handle, self._dtype)
```

# 梯度更新

* Variable's grad update

```python
class GradientDescentOptimizer(optimizer.Optimizer):
  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)
```

```cpp
REGISTER_OP("ApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyGradientDescentShapeFn);
    
REGISTER_OP("ResourceApplyGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("delta: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyGradientDescentShapeFn);
```
