

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
class HashEmbeddingVar : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data) = 0;
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func) = 0;
private:
  std::unordered_map<T, Tensor> tensor_map_;
};

// 用经典的hash方法，数组的方式实现
// 给定一个key, hash到一个slot中去，在当前slot中查找key，如果查中则返回，如果查不中，则slot + 1，依次循环，直到遇到empty_key为止
class DenseEmbeddingVar : public LookupableEmbeddingVar {
public:
  virtual void GetEmbedding(T key, float** data);
  virtual void PutEmbedding(T key, float* tensor, UpdateFunc func);
private:
  PersistentTensor keys_;
  PersistentTensor values_;
};

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

