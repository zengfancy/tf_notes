
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

template <typename K, typename V>
class DenseEmbeddingVar2 : public EmbeddingVar<K, V> {
public:
  
private:
  PersistentTensor first_keys_;
  PersistentTensor keys_;
  PersistentTensor values_;
  PersistentTensor next_;
};
