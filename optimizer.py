

class _EmbResourceVariableProcessor(_OptimizableVariable):
  def update_op(self, optimizer, g):
    if not isinstance(g, ops.IndexedSlices):
      raise NotSupportError("")
      
    return optimizer._emb_resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices)
          
class Optimizer:
  def _emb_resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    raise NotImplementedError()
    
class GradientDescentOptimizer(optimizer.Optimizer):
  def _emb_resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    # import from REGISTER_OP
    return gen_embedding_var_ops.embedding_scatter_add(handle, indices, grad)
