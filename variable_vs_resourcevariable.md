
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

```
