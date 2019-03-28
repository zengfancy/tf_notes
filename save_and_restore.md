* 可参考https://tensorflow.google.cn/guide/saved_model , 里面讲述了导出到 checkpoint 和 saved model

* 导出模型到Checkpoint

```cpp
class SaveOp : public OpKernel {
  void Compute(OpKernelContext* context) override {
    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, false);
  }
};
```

```proto
message SaverDef {
  // The name of the tensor in which to specify the filename when saving or
  // restoring a model checkpoint.
  string filename_tensor_name = 1;

  // The operation to run when saving a model checkpoint.
  string save_tensor_name = 2;

  // The operation to run when restoring a model checkpoint.
  string restore_op_name = 3;
}
```

```python
# saver.py
# 这个文件的功能主要是save/restore to/from checkpoint，保存的时候不仅保持变量，也保存meta graph (调用export_meta_graph函数），
# restore 的时候，先从meta graph里面 import graph, 然后 restore variables

class BaseSaverBuilder:
  def save_op(self, filename_tensor, saveables):
    for saveable in saveables:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)
    return io_ops._save(
          filename=filename_tensor,
          tensor_names=tensor_names,
          tensors=tensors,
          tensor_slices=tensor_slices)
          
  def _AddSaveOps(self, filename_tensor, saveables):
    save = self.save_op(filename_tensor, saveables)
    return control_flow_ops.with_dependencies([save], filename_tensor)
    
  def _AddRestoreOps(self,
                     filename_tensor,
                     saveables):
    for saveable in saveables:
      tensors = self.restore_op(
                filename_tensor, saveable, preferred_shard)
      assign_ops.append(saveable.restore(tensors, shapes))
    return control_flow_ops.group(*assign_ops, name=name)
    
  def _build_internal(self, names_to_saveables):
    saveables = saveable_object_util.validate_and_slice_inputs(
        names_to_saveables)
    save_tensor = self._AddSaveOps(filename_tensor, saveables)
    restore_op = self._AddRestoreOps(filename_tensor, saveables,
                                           restore_sequentially, reshape)
    return saver_pb2.SaverDef(
          filename_tensor_name=filename_tensor.name,
          save_tensor_name=save_tensor.name,
          restore_op_name=restore_op.name,
          max_to_keep=max_to_keep,
          sharded=sharded,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
          version=self._write_version)
          

# use for Variable
class ReferenceVariableSaveable(saveable_object.SaveableObject):

# use for ResourceVariable
class ResourceVariableSaveable(saveable_object.SaveableObject):

# Convert a variable, operation, or SaveableObject to a SaveableObject.
def saveable_objects_for_op(op, name):

def validate_and_slice_inputs(names_to_saveables):
    for converted_saveable_object in saveable_objects_for_op(op, name):
      _add_saveable(saveables, seen_ops, converted_saveable_object)
      
class Saver:
  def __init__:
    if self._var_list is None:
      # pylint: disable=protected-access
      self._var_list = variables._all_saveable_objects()
    if not self.saver_def:
        self.saver_def = self._builder._build_internal(  # pylint: disable=protected-access
          self._var_list,
          reshape=self._reshape,
          sharded=self._sharded,
          max_to_keep=self._max_to_keep,
          keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
          name=self._name,
          restore_sequentially=self._restore_sequentially,
          filename=checkpoint_path,
          build_save=build_save, build_restore=build_restore)
      
  def save:
    sess.run(
      self.saver_def.save_tensor_name,
      {self.saver_def.filename_tensor_name: checkpoint_file})
        
def _all_saveable_objects(scope=None):
  return (ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope) +
    ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS, scope))
```


* 导出模型到 SavedModel : 也是将 variable 导出到 一个单独的文件，和上面差不多，最终还是调用了 Saver 的接口，只不过没有导出 checkpoints 状态而已

```python
def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
  signature_def_map = {
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          signature_def_utils.predict_signature_def(inputs, outputs)
  }
  b = builder.SavedModelBuilder(export_dir)
  b.add_meta_graph_and_variables(
      session,
      tags=[tag_constants.SERVING],
      signature_def_map=signature_def_map,
      assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      legacy_init_op=legacy_init_op,
      clear_devices=True)
  b.save()
  
class SavedModelBuilder(object):
  def add_meta_graph_and_variables(self,
                                   sess,
                                   tags,
                                   signature_def_map=None,
                                   assets_collection=None,
                                   legacy_init_op=None,
                                   clear_devices=False,
                                   main_op=None):
    saver = tf_saver.Saver(
        variables._all_saveable_objects(),  # pylint: disable=protected-access
        sharded=True,
        write_version=saver_pb2.SaverDef.V2,
        allow_empty=True)
    saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
    meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices)
    # will modify self._saved_model
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
    
  def add_meta_graph(self,
                     tags,
                     signature_def_map=None,
                     assets_collection=None,
                     legacy_init_op=None,
                     clear_devices=False,
                     main_op=None):
    saver = tf_saver.Saver(
        variables._all_saveable_objects(),  # pylint: disable=protected-access
        sharded=True,
        write_version=saver_pb2.SaverDef.V2,
        allow_empty=True)
    meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices)
    # will modify self._saved_model
    self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
    
  def save(self, as_text=False):
    file_io.write_string_to_file(path, str(self._saved_model))
```

* 导出到 proto : tensorflow 中导出 variable 的时候是将它转换成一个 constant op 导出的。阿里的做法只有导出到 odps 表格，没有实现导出到 proto 模型的做法
