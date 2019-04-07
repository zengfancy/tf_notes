

REGISTER_OP("EmbeddingVarHanleOp")
  .Attr("container: string = ''")
  .Attr("dtype: type")
  .Attr("K: {int64, int32, string}")
  .Attr("shape: shape")
  .Output("resource: resource")
  .SetIsStateful()
  .SetShapeFn([] (InferenceContext* ctx) {
      })
  .Doc(R"()");

REGISTER_OP("InitializeEmbeddingVarOp")
  .Input("resource: resource")
  .Input("value: dtype")
  .Input("empty_key: K")
  .Attr("dtype: type")
  .Attr("K: {int64, int32, string}")
  .Attr("shape: shape")
  .Attr("init_random: bool = true")
  .SetShapeFn([] (InferenceContext* ctx) {
      })
  .Doc(R"()");

REGISTER_OP("EmbeddingLookupOp")
  .Input("resource: resource")
  .Input("indices: K")
  .Output("output: dtype")
  .Attr("dtype: type")
  .Attr("K: {int64, int32, string}")
  .SetShapeFn([] (InferenceContext* ctx) {
      })
  .Doc(R"()");

