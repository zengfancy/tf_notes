
template <typename K, typename V>
class EmbeddingLookupOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    const ResourceHandle& handle = HandleFromInput(context, 0);
    EmbeddingVar<K, V>* var = nullptr;
    auto status = LookupResource(context, handle, &var);
    OP_REQUIERES(status.ok())...

    const Tensor& key = context->input(1);
    const int64 num_elems = key.NumElements();

    TensorShape outShape = key.shape();
    outShape.AppendShape(var->GetValueShape());
	  
    Tensor* value = nullptr;
    status = context->allocate_output(0, outShape, &value);

    const auto emb_len = var->GetEmbLen();
    const auto key_vec = key.shaped<K, 1>({num_elems});
    auto val_matrix = value->shaped<V, 2>({num_elems, emb_len});

    for (int64 i=0; i<num_elems; ++i) {
      V* data = nullptr;
      var->GetEmbedding(key_vec(i), &data);
      memcpy(&val_matrix(i, 0), data, emb_len * sizeof(V));
    }
  }
};


template <typename K, typename V, scatter_op::UpdateOp op>
class EmbeddingUpdateOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    const ResourceHandle& handle = HandleFromInput(context, 0);
    EmbeddingVar<K, V>* var = nullptr;
    auto status = LookupResource(context, handle, &var);
    OP_REQUIERES(status.ok())...

    const Tensor& key = context->input(1);
    const Tensor& delta = context->input(2);

    const int64 num_elems = key.NumElements();
    const auto emb_len = var->GetEmbLen();
    DCHECK(delta.NumElements() == num_elems * emb_len);

    const auto key_vec = key.shaped<K, 1>({num_elems});
    auto delta_matrix = delta.shaped<V, 2>({num_elems, emb_len});

    for (int64 i=0; i<num_elems; ++i) {
      var->PutEmbedding(key_vec(i), &delta_matrix(i, 0), op);
    }
  }
};

template <typename K, typename V>
EmbeddingVar<K, V>* NewEmbeddingVar() {
}

template <typename K, typename V>
class InitializeEmbeddingOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
      errors::InvalidArgument(
      "Variable and value dtypes don't match; respectively, ",
      dtype_, " and ", context->input(1).dtype()));
    DHMVar<K, V>* variable = nullptr;
    OP_REQUIRES_OK(
      context,
      LookupOrCreateResource<EmbeddingVar<K, V>>(
        context, HandleFromInput(context, 0), &variable,
          [](EmbeddingVar<K, V>** ptr) {
	    *ptr = NewEmbeddingVar<K, V>();
	    return Status::OK();
	  }
      )
    );
  }
};

	
// check一下是否tensorflow已经实现了这个算子	
class EmbeddingKeyDedupOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
  }
};

// check一下是否tensorflow已经实现了这个算子	
class EmbeddingKeyDuplicateOp : public OpKernel {
  void Compute(OpKernelContext* context) {
  }
};

// check一下是否tensorflow已经实现了这个算子	
class EmbeddingGradReduceOp : public OpKernel {
  void Compute(OpKernelContext* context) {
  }
};


class ImportEmbeddingOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    const ResourceHandle& handle = HandleFromInput(context, 0);
    const Tensor& key = context->input(1);
    const Tensor& emb_value = context->input(2);
  }
};

class ExportEmbeddingOp : public OpKernel {
public:
  void Compute(OpKernelContext* context) {
    const ResourceHandle& handle = HandleFromInput(context, 0);
    Tensor* key = nullptr;
    Tensor* emb_value = nullptr;
    status = context->allocate_output(0, shape, &key);
    status = context->allocate_output(1, shape, &emb_value);
  }
};
	
	
#define REGISTER_KERNELS(K, V) \
  REGISTER_KERNEL_BUILDER(Name("InitializeEmbeddingOp") \
      .Device(DEVICE_CPU)  \
      .TypeConstraint<K>("TKeys")  \
      .TypeConstraint<V>("TVals"), \
      InitializeEmbeddingOp<K, V>); 

#define REGISTER_KERNELS_K(V)  \
  REGISTER_KERNELS(int64, V) \
  REGISTER_KERNELS(int32, V) \
  REGISTER_KERNELS(string, V)


TF_CALL_ALL_TYPES(REGISTER_KERNELS_K);
