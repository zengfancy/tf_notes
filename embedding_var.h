
template <typename K, typename V>
class EmbeddingVar : public ResourceBase {
public:
  virtual void GetEmbedding(K key, V** data) = 0;
  virtual void PutEmbedding(K key, const V* data, int64 len, scatter_op::UpdateOp op) = 0;
  virtual void DeleteKey(K key) = 0;
  
  TensorShape GetEmbShape();
  int64 GetEmbLen();
};

template <typename K, typename V>
class HashEmbeddingVar : public EmbeddingVar<K, V> {

private:
  std::unordered_map<K, PersistentTensor> tensor_map_;
};

/* 
  hash bucket function : hash(key) % buckets(take buckets as 10, for example)
  take key = "fanxi" as example. if hash("fanxi") = 4098, then the bucket is 4098 % 10 = 8.
*/

/* 1. the length of start array is as long as the all hash buckets.
   2. the length of key, value, next array is the same. 
   3. factor_ : when keys'num > buckets_ * facotr_ * 1.1, the array should Grow()
                when keys'num < buckets_ * facotr_ * 0.9, the array should Shrink()
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
  
private:
  void Shrink();
  void Grow();
private:
  PersistentTensor first_keys_;
  PersistentTensor keys_;
  PersistentTensor values_;
  PersistentTensor next_;
  
  int64 buckets_;
  int32 factor_;
};


