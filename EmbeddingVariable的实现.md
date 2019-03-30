
## OpKernel 的实现

* Tensorflow 对变量的实现有两种方式，Variable和ResourceVariable, 其中ResourceVariable更加安全，也是TF2.0之后的默认使用方式
* 基于以上原因，EmbeddingVariable 的 OpKernel 实现完全参考了 ResourceVariable 的 OpKernel实现
* 以下为虚基类 LookupableEmbeddingVar 的定义

```cpp
class LookupableEmbeddingVar {
public:
  // 查找对应 key 的 tensor, 如果为空，则插入一个 default tensor, 并返回
  virtual void GetEmbedding(T key, float** data) = 0;
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func) = 0;
  
  int64_t emb_len_;
};
```
* 有三种实现方式，各有优缺点
  + 内部结构采用std::unordered_map
  + 内部结构采用经典的2数组 HashSlot 方式
  + 内部结构采用非经典的4数组 ArrayHashMap 方式
  
```cpp
// 直接用unordered map
// 缺点：Tensor过多，内存分布比较散乱，内存cache性能差
class HashEmbeddingVar : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data) = 0;
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func) = 0;
private:
  std::unordered_map<T, Tensor> tensor_map_;
};

// 1. 用经典的hash方法，数组的方式实现，两个数组，一个key数组，一个value数组
// 2. 给定一个key, hash到一个slot中去，在当前slot中查找key，如果查中则返回，如果查不中，则slot += 1，依次循环，直到遇到empty_key为止
// 3. hash数组的 capacity 是动态增长的，一般来说 capacity 应该为 2^n - 1, 当 capacity < length * factor （比如factor = 0.3) 时，
//    capacity应该倍增
// 4. 这种结构的 value 数组的长度必须跟 keys 数组的长度一致，有浪费空间之嫌
class DenseEmbeddingVar : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data);
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func);
private:
  PersistentTensor keys_;
  PersistentTensor values_;
};

// 1. ArrayHashMap，使用4个数组，借鉴了C#里面的Dictionary实现，详细原理可见我上次分享的PPT
// 2. 优势：内存使用效率高，访问速度也很快，value和key长度的真实长度就是它的实际长度，所以节省内存
// 3. first_keys_ 长度应该动态增长，当增长（一般为倍增）时，next_ 数组需要重新初始化
class DenseEmbeddingVar2 : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data);
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func);
private:
  PersistentTensor first_keys_;
  PersistentTensor keys_;
  PersistentTensor values_;
  PersistentTensor next_;
};

```
* EmbeddingVarHandleOp
  + 类似于VarHandleOp, 输出一个handle
* EmbeddingInitOp
  + 创建一个EmbeddingVariable, 类似于AssignVariableOp
  
* EmbeddingLookupOp & EmbeddingUpdateOp 的单机实现

```cpp
         ^
         |  val_tensor
         |
    EmbeddingLookUpOp   < ----  handle
         ^
         |  key_tensor
         |
   
   
// no output for update


         EmbeddingUpdateOp   < ----  handle
             ^     ^
 key_tensor  |     | val_tensor
             |     |
```

```cpp
// 查找Op, 对于每个key, 都返回一个对应的 embedding vector
class EmbeddingLookupOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    Tensor* in_tensor = context->input(0);
    Tensor* out_tensor = context->allocate_output(0);
    
    foreach elem in in_tensor {
      float* emb;
      emb_var_->GetEmbedding(elem, &emb);
      Copy(&out_tensor(i), emb, emb_len);
    }
  }
};

template <class UpdateFunc>
class EmbeddingUpdateOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    Tensor* key_tensor = context->input(0);
    Tensor* val_tensor = context->input(1);
    foreach <key:val> {
      emb_var_->PutEmbedding(key.scalar(), val.vec(), update_func);
    }
  }
};
REGISTER_KERNEL_BUILDER("EmbeddingScatterAdd")...EmbeddingUpdateOp<Add>...
REGISTER_KERNEL_BUILDER("EmbeddingScatterAssign")...EmbeddingUpdateOp<Assign>...
REGISTER_KERNEL_BUILDER("EmbeddingScatterSub")...EmbeddingUpdateOp<Sub>...
```

* EmbeddingLookup & EmbeddingUpdate 的分布式PS实现
  + EmbeddingKeyDedupOp : 将 key 去重, Tensorflow似乎已经有实现
  + EmbeddingDuplicateOp : 反去重, Tensorflow似乎已经有实现
  + EmbeddingLookUpOp : 和单机版的 EmbeddingLookUpOp 一样
  + EmbeddingGradReduceOp : 将 key 去重，并 reduce grad value, Tensorflow似乎已经有实现
  + EmbeddingUpdateOp : 和单机版的 EmbeddingUpdateOp 一样
  
```cpp
         ^                                                               ^   
         |  val_tensor    ----------------------------                   |  val_tensor_2
         |                                            |                  |
    EmbeddingLookUpOp                                 --------> EmbeddingDuplicateOp
         ^                                                           ^      ^
         |                                                           |      |
         |  key_tensor_   --------------------------------------------      |
         |                                                                  |
   EmbeddingKeyDedupOp                                                      |
         ^                                                                  |
         |  key_tensor   ----------------------------------------------------
         |
   
// no output for update

         EmbeddingUpdateOp
             ^     ^
 key_tensor_ |     | val_tensor_
             |     |
        EmbeddingGradReduceOp
             ^     ^
 key_tensor  |     | val_tensor
             |     |        
```

```cpp
class EmbeddingKeyDedupOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
  }
};

class EmbeddingDuplicateOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
  }
};

class EmbeddingGradReduceOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
  }
};
```

## 导入导出OpKernel的实现

* Import & Export

```cpp
             ^     ^
 key_tensor  |     | val_tensor
             |     |
        ExportEmbeddingOp
        
        
        ImportEmbeddingOp
             ^     ^
 key_tensor  |     | val_tensor
             |     |      
             
class ImportEmbeddingOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
  }
};

class ExportEmbeddingOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
  }
};
```

## Ops 的实现

* 定义如下Ops:
  + EmbeddingVarHandleOp : 
  + EmbeddingGather ：查找，输入key，返回value
  + EmbeddingScatterAdd/Sub/Assign ：更新, 输入key, grad，无返回, 这个Op用于SGD算法的梯度更新，Adam, AdaGrad等复杂更新算法需要额外的Op才能完成
  + EmbeddingKeyDedup ：key去重，输入key, 返回去重后的key
  + EmbeddingDuplicate ：key, value反去重，输入key1, key2, value, 返回duplicate之后的value
  + EmbeddingGradReduce ：key去重，并将grad合并,输入key, grad,返回合并后的key, grad
  + ExportEmbedding ：导出
  + ImportEmbedding ：导入

* 会产生一个python 文件 gen_kv_embedding_ops.py

```python
def embedding_var_handle_op:
def init_embedding_var:
def embedding_gather:
def embedding_scatter_add:
def embedding_scatter_sub:
def embedding_scatter_assign:
def embedding_key_dedup:
def embedding_duplicate:
def embedding_grad_reduce:
def export_embedding:
def import_embedding:
```

## Python 层的实现
* 有关 Variable, ResourceVariable 的实现可以参考 https://github.com/zengfancy/tf_notes/blob/master/Variable_vs_Resourcevariable.md
* EmbeddingVariable, 参考ResourceVariable实现
  + 继承自VariableV1
  + 实现其init_from_proto, init_from_args
  + lookup 逻辑如 sparse_read
  
```python
class EmbeddingVariable(VariableV1):
  def __init__:
    self._handle = gen_kv_embedding_ops.embedding_var_hanle_op()
    self._init_op = gen_kv_embedding_ops.init_embedding_var(self._handle, emb_shape)
    
  def sparse_read(indices):
    return gen_kv_embedding_ops.embedding_lookup(self._handle, indices)
```
  
* 修改variable_scope.py，支持从get_variable()函数得到一个EmbeddingVariable
  
* Tensorflow 里面 embedding_lookup 函数调用了 array_ops.gather 函数，需要修改 gather 这个函数，普通的 Variable 调用 gen_array_ops.gather_v2，EmbeddingVariable 应该调用 embedding_gather

## 导入导出相关
* Variable, ResourceVariable 的导入导出可以参考 https://github.com/zengfancy/tf_notes/blob/master/save_and_restore.md
* 仿照ResourceVariableSaveable的做法，实现一个EmbeddingVariableSaveable的类即可
  
```python
class EmbeddingVariableSaveable(saveable_object.SaveableObject):
# 实现 save 逻辑
def __init__(self, var, slice_spec, name):
  def _read_emb_variable(v):
    def f():
      with ops.device(v.device):
        key, value = gen_kv_embedding_ops.export_embedding(v.handle)
        # To allow variables placed on non-CPU devices to be checkpointed,
        # we copy them to CPU on the same machine first.
        with ops.device("/device:CPU:0"):
          return array_ops.identity(key), array_ops.identity(value)
    return f
  key_tensor, val_tensor = _read_emb_variable(var)
  spec_key = saveable_object.SaveSpec(key_tensor, slice_spec, name + "_key",
                                    dtype=var.dtype)
  spec_val = saveable_object.SaveSpec(tensor, slice_spec, name + "_value",
                                    dtype=var.dtype)                                  
  super(EmbeddingVariableSaveable, self).__init__(var, [spec_key, spec_value], name)

# 实现 restore 逻辑
def restore(self, restored_tensors, restored_shapes):

```

## 梯度更新相关
* 有关 Variable, ResourceVariable 的梯度更新可以参考 https://github.com/zengfancy/tf_notes/blob/master/Variable_vs_Resourcevariable.md, 下一步

```python
class _DenseResourceVariableProcessor(_OptimizableVariable):
  def update_op(self, optimizer, g):
    if isinstance(g, ops.IndexedSlices):
      return optimizer._resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices)
    else:
      return optimizer._resource_apply_dense(g, self._v)
```

* 添加一个名叫 KvResourceVariableProcessor 的 _OptimizerVariable
```python
class _KvResourceVariableProcessor(_OptimizableVariable):
  def update_op(self, optimizer, g):
    if not isinstance(g, ops.IndexedSlices):
      raise NotSupportError("")
      
    return optimizer._kv_resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices)
```

* 给 Optimizer 基类添加两个函数 _kv_resource_apply_sparse_duplicate_indices 与 _kv_resource_apply_sparse，默认不实现，由子类来实现，以sgd为例
```python
class GradientDescentOptimizer(optimizer.Optimizer):
  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return kv_resource_variable_ops.embedding_scatter_add(handle, indices, grad）
```

## tensorflow serving相关
* 前期可仿照 ConstantOp 实现一个 ContantEmbeddingOp, ContantEmbeddingOp 的 lookup 逻辑跟EmbeddingLookupOp类似，不需要有 Update 的功能。save的时候将key, value存储到 proto 当中，后期如果占内存太大，超过4G不允许，可考虑将 key, value 保存到一个单独的文件中，ConstantOp 在初始化的时候从文件中初始化

## EmbeddingVariable 梯度更新跟 RingAllReduce 框架如何结合
* Embedding特征随机初始化不一致的问题 ：修改随机初始化的规则，使得初始化不随机，比如：根据特征key的值，算出一个固定的hash值，再根据hash值确定初始化值，注意数学分布符合高斯分布即可
* 梯度 reduce 的问题 ：应该修改一下 Hovorod 的优化器就可以，使用hovorod训练代码示例如下
  + hvd.DistributedOptimizer 的作用是包装了adam optimizer, 并将分布式梯度 gather, reduce，需要修改 DistributedOptimizer 以适应 EmbeddingVariable
  + BroadcastGlobalVariablesHook 的作用是随机初始化 variable 的时候保持每个机器上的初始化值一致，EmbeddingVariable 初始化的时候是空的，所以应该不需要关心
  
```python
import tensorflow as tf
import horovod.tensorflow as hvd

# 1、初始化 Horovod
hvd.init()

# 2、使用GPU来处理本地队列（向每个TensorFlow进分配一个进程）
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# 3、建立模型
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

# 4、添加Horovod分布式优化器
opt = hvd.DistributedOptimizer(opt)

# 5、Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)

```
  
