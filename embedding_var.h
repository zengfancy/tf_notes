
template <typename K, typename V>
class EmbeddingVar : public ResourceBase {
public:
  virtual void GetEmbedding(K key, V** data) = 0;
  virtual void PutEmbedding(K key, V* data, scatter_op::UpdateOp op) = 0;
  
  TensorShape GetValueShape();
};

template <typename K, typename V>
class HashEmbeddingVar : public EmbeddingVar<K, V> {

private:
  std::unordered_map<K, PersistentTensor> tensor_map_;
};

/*
start array : -1   1  3  0  -1  -1  5   7  -1  -1
key   array :  3  21  13 42 53  76  43  97 86
value array :  v   v  v  v   v   v  v   v   v
next  array :  2  -1  4  -1  6   8  -1  -1  -1
*/

/*
-1 -1 -1 0 -1 -1 -1 -1 -1  1
3   9
v   v
-1 -1
*/
template <typename K, typename V>
class DenseEmbeddingVar2 : public EmbeddingVar<K, V> {
public:
  
private:
  PersistentTensor first_keys_;
  PersistentTensor keys_;
  PersistentTensor values_;
  PersistentTensor next_;
};


