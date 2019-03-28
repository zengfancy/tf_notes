
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
