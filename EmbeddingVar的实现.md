
## OpKernel 的实现

* EmbeddingVariable的OpKernel实现ResourceVariable 的 OpKernel实现
* LookupableEmbeddingVar 的定义

```cpp
class LookupableEmbeddingVar {
public:
  // 查找对应 key 的 tensor, 如果为空，则插入一个 default tensor, 并返回
  virtual void GetEmbedding(T key, float** data) = 0;
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func) = 0;
  
  int64_t emb_len_;
};

// 直接用unordered map
// 缺点：内存散乱，访问性能差
class HashEmbeddingVar : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data) = 0;
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func) = 0;
private:
  std::unordered_map<T, Tensor> tensor_map_;
};

// 用经典的hash方法，数组的方式实现
// 给定一个key, hash到一个slot中去，在当前slot中查找key，如果查中则返回，如果查不中，则slot + 1，依次循环，直到遇到empty_key为止
// hash数组的 capacity 是动态增长的，一般来说 capacity 应该为 2^n - 1, 当 capacity < length * factor 时，capacity应该倍增
// 但是这种结构的 value 数组的长度必须跟 keys 数组的长度一致，有浪费空间之嫌
class DenseEmbeddingVar : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data);
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func);
private:
  PersistentTensor keys_;
  PersistentTensor values_;
};

// 优势：内存使用效率高，访问速度也很快
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

* EmbeddingLookupOp & EmbeddingUpdateOp 的单机实现

```cpp
         ^
         |  val_tensor
         |
    EmbeddingLookUpOp 
         ^
         |  key_tensor
         |
   
   
// no output for update

         EmbeddingUpdateOp
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
```

* EmbeddingLookup & EmbeddingUpdate 的分布式PS实现
  + EmbeddingKeyDedupOp : 将 key 去重
  + EmbeddingDuplicateOp : 反去重
  + EmbeddingLookUpOp : 和单机版的 EmbeddingLookUpOp 一样
  + EmbeddingGradReduceOp : 将 key 去重，并 reduce grad value
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

* EmbeddingLookup ：查找，输入key，返回value
* EmbeddingUpdate ：更新, 输入key, grad，无返回, 这个Op用于SGD算法的梯度更新，Adam, AdaGrad等复杂更新算法需要额外的Op才能完成
* EmbeddingKeyDedup ：key去重，输入key, 返回去重后的key
* EmbeddingDuplicate ：key, value反去重，输入key1, key2, value, 返回duplicate之后的value
* EmbeddingGradReduce ：key去重，并将grad合并,输入key, grad,返回合并后的key, grad
* ExportEmbedding ：导出
* ImportEmbedding ：导入

## Python 层的实现
* 有关 Variable, ResourceVariable 的实现可以参考 https://github.com/zengfancy/tf_notes/blob/master/Variable_vs_Resourcevariable.md
* EmbeddingVariable
  + 从ResourceVariable继承
  + 实现其init_from_proto, init_from_args
  + 其他逻辑如 read_value, read, sparse_read 参考 ResourceVariable的实现
  
* 修改variable_scope.py，支持从get_variable()函数得到一个EmbeddingVariable
  
* 修改embedding_lookup函数，复用其中的逻辑

* 导入导出相关
  + Variable, ResourceVariable 的导入导出可以参考 https://github.com/zengfancy/tf_notes/blob/master/save_and_restore.md
  + 仿照ResourceVariableSaveable的做法，实现一个EmbeddingVariableSaveable的类即可

## 梯度更新相关
* 有关 Variable, ResourceVariable 的梯度更新可以参考 https://github.com/zengfancy/tf_notes/blob/master/Variable_vs_Resourcevariable.md
* 添加一个名叫 KvResourceVariableProcessor 的 _OptimizerVariable
* 给 Optimizer 添加两个函数 _kv_resource_apply_sparse_duplicate_indices 与 _resource_apply_sparse，默认不实现，由子类来实现

## tensorflow serving相关
* 前期可仿照 ConstantOp 实现一个 ContantEmbeddingOp, 将key, value存储到 proto 当中，后期如果占内存太大，超过4G不允许，可考虑将 key, value 保存到一个单独的文件中，ConstantOp 在初始化的时候从文件中初始化
