?	???5@???5@!???5@	Y?s?9??Y?s?9??!Y?s?9??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???5@??1 ?3@1??????A??D????I???|???Y??:???*	533333I@2F
Iterator::ModelǺ?????!??8??8F@)?? ?rh??1?u]?u?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C??6??!Y?eY?e9@)Ǻ?????1??8??86@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{?G?z??!v]?u]?3@)?<,Ԛ?}?1??}??,@:Preprocessing2U
Iterator::Model::ParallelMapV2??_vOv?!۶m۶m%@)??_vOv?1۶m۶m%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy?&1???!r?q?K@)??H?}m?1$I?$I?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOf?!۶m۶m@)??_vOf?1۶m۶m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!AA7@)-C??6Z?1Y?eY?e	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6Z?!Y?eY?e	@)-C??6Z?1Y?eY?e	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 90.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9X?s?9??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??1 ?3@??1 ?3@!??1 ?3@      ??!       "	????????????!??????*      ??!       2	??D??????D????!??D????:	???|??????|???!???|???B      ??!       J	??:?????:???!??:???R      ??!       Z	??:?????:???!??:???JGPUYX?s?9??b ?"E
)gradient_tape/sequential_1/dense_5/MatMulMatMul[?v?Il??![?v?Il??"-
IteratorGetNext/_1_Send?<i?%??!??Y$?H??"G
+gradient_tape/sequential_1/dense_6/MatMul_1MatMul???",???!Y#ך>	??"G
+gradient_tape/sequential_1/dense_7/MatMul_1MatMul.oa?t??!$??f??"G
+gradient_tape/sequential_1/dense_8/MatMul_1MatMulY?G?b??!?c???m??"E
)gradient_tape/sequential_1/dense_7/MatMulMatMulrI??????!?6LrE???"E
)gradient_tape/sequential_1/dense_6/MatMulMatMul??K-??!???????"7
sequential_1/dense_5/MatMulMatMul??K-??!J???z??"7
sequential_1/dense_7/MatMulMatMul??K-??!?H`p?]??"7
sequential_1/dense_8/MatMulMatMul??K-??! ?ņ@??Q      Y@Y      (@a      V@q,V?aO@y?_0?qf??"?
both?Your program is POTENTIALLY input-bound because 90.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?62.1436% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 