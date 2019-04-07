

class EmbeddingVariable(variables.VariableV1):
  """
  initial_value: it can't be None, as it provide the shape of the embedding
  dtype: if not None, initial_value will be transformed to dtype
  init_random: if not None, the variable will only the shape and dtype of initial_value
  """
  def __init__(self,
          initial_value=None,
          init_random=True,
          trainable=True,
          collections=None,
          dtype=None,
          name=None,
          invalid_key=None):
    if not context.in_graph_mode():
      raise ValueError("")
    if initial_value is None:
      raise ValueError("")

    self._init_from_args(
            name=name,
            dtype=dtype,
            trainable=trainable,
            collections=collections,
            initial_value=initial_value,
            init_random=init_random,
            invalid_key=invalid_key)


  def _init_from_args(self,
          name=None,
          dtype=None,
          trainable=True,
          collections=None,
          initial_value=None,
          init_random=True,
          invalid_key=-1):
    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]

    self._trainable = trainable
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

    if callable(initial_value):
      initial_value = ops.convert_to_tensor(
              initial_value(), name="initial_value", dtype=dtype)
    else:
      initial_value = ops.convert_to_tensor(
              initial_value, name="initial_value", dtype=dtype)

    self._invalid_key = invalid_key
    self._key_type = ops.convert_to_tensor(invalid_key,
           name="invalid_key", preferred_dtype=dtypes.int64).dtype.base_dtype

    self._handle = self._embedding_variable_handle(
            shape=initial_value.get_shape(),
            dtype=initial_value.dtype.base_dtype,
            ktype=self._key_type,
            name=name)

    if initial_value is not None:
      with ops.name_scope("Initialize") as n, ops.colocate_with(self._handle):
        self._initializer_op = gen_embedding_var_ops.initialize_embedding_var_op(
          self._handle,
          self._build_initializer_expr(initial_value),
          empty_key=ops.convert_to_tensor(invalid_key, prefered_type=dytpe.int64),
          shape=initial_value.get_shape(),
          dtype=initial_value.dtype.base_dtype,
          K=self._key_type,
          init_random=init_random,
          name=n)

  def trainable(self):
    return self._trainable
          
  def key_type(self):
    return self._key_type

  def _embedding_variable_handle(self, shape, dtype, ktype, name):
    container = ops.get_default_graph()._container  # pylint: disable=protected-access
    if container is None:
      container = ""
    return gen_embedding_var_ops.embedding_var_handle_op(shape=shape, 
             dtype=dtype,
             name=name,
             K=ktype,
             container=container)


  def initializer(self):
    return self._initializer_op

  def handle(self):
    return self._handle

  def sparse_read(self, indices, name=None):
    """
    This function is like ResourceVariable.sparse_read.
    See also array_ops.gather. 
    """
    return gen_embedding_var_ops.embedding_lookup_op(self._handle, indices, name=name)

