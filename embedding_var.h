
template <typename K, typename V>
class EmbeddingVar : public ResourceBase {
public:
  EmbeddingVar((OpKernelContext* ctx, OpKernel* kernel);
  
  virtual void GetEmbedding(K key, V** data) = 0;
  virtual void PutEmbedding(K key, const V* data, int64 len, scatter_op::UpdateOp op) = 0;
  virtual void DeleteKey(K key) = 0;
  virtual void Initialize(OpKernelContext* ctx) = 0;
  
  TensorShape GetEmbShape();
  int64 GetEmbLen();
  
protected:
  TensorShape emb_shape_;
};

template <typename K, typename V>
class HashEmbeddingVar : public EmbeddingVar<K, V> {

private:
  std::unordered_map<K, PersistentTensor> tensor_map_;
};

/*
  It's the standard implementation of HashTable as an array.
  
  hash bucket function : hash(key) % buckets(take buckets as 100, for example)
  take key = "fanxi" as example. if hash("fanxi") = 4098, then the bucket is 4098 % 100 = 98.
  let's check whether slot(98) valued "fanxi", if not, check slot(99)...
  if invalid_key_ found, then we say this hashmap doesn't have a key valued "fanxi"
  
  3. factor_ : when keys'num > buckets_ * factor_ * 1.1, the array should Grow()
               when keys'num < buckets_ * factor_ * 0.9, the array should Shrink()
*/
template <typename K, typename V>
class DenseEmbeddingVar1 : public EmbeddingVar<K, V> {
public:
  virtual void GetEmbedding(K key, V** data);
  virtual void PutEmbedding(K key, const V* data, int64 len, scatter_op::UpdateOp op);
  virtual void DeleteKey(K key);
  virtual void Initialize(OpKernelContext* ctx);
  
private:
  void Shrink();
  void Grow();
  
  PersistentTensor keys_;
  PersistentTensor values_;
  
  float factor_; // valued 0.3, for example
  int64 buckets_; // usually valued 2^n
  K invalid_key_;
};

/* 
  hash bucket function : hash(key) % buckets(take buckets as 10, for example)
  take key = "fanxi" as example. if hash("fanxi") = 4098, then the bucket is 4098 % 10 = 8.
*/

/* 1. the length of start array is as long as the all hash buckets.
   2. the length of key, value, next array is the same. 
   3. factor_ : when keys'num > buckets_ * factor_ * 1.1, the array should Grow()
                when keys'num < buckets_ * factor_ * 0.9, the array should Shrink()
*/

/*
start array : -1   1  3  0  -1  -1  5   7  -1  -1
key   array :  3  21  13 42 53  76  43  97 86
value array :  v   v  v  v   v   v  v   v   v
next  array :  2  -1  4  -1  6   8  -1  -1  -1
*/

/*
start array : -1 -1 -1 0 -1 -1 -1 -1 -1  1
key   array : 3   9
value array : v   v
next  array : -1 -1
*/
template <typename K, typename V>
class DenseEmbeddingVar2 : public EmbeddingVar<K, V> {
public:
  virtual void GetEmbedding(K key, V** data);
  virtual void PutEmbedding(K key, const V* data, int64 len, scatter_op::UpdateOp op);
  virtual void DeleteKey(K key);
  virtual void Initialize(OpKernelContext* ctx) {
    DataType key_type = DataTypeToEnum<K>::v();
    DataType value_type = DataTypeToEnum<V>::v();
    
    ctx->allocate_persistent(dt, shape...)
  }
  
private:
  void Shrink();
  void Grow();
private:
  PersistentTensor first_keys_;
  PersistentTensor keys_;
  PersistentTensor values_;
  PersistentTensor next_;
  
  int64 buckets_;   // usually valued 2^n
  int64 len_;
  int64 capacity_;
  float factor_;    // valued 2.0, for example
};

template <typename K, typename V>
EmbeddingVar<K, V>* NewEmbeddingVar() {
}

