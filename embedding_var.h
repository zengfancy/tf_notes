
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
  
};

