?	Y??+??6@Y??+??6@!Y??+??6@	g??????g??????!g??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Y??+??6@+mq??,4@1/?>:u???A??(\?¥?I???!???Y?Y.????*	23333sQ@2F
Iterator::Modelh??|?5??!}N???!E@) ?o_Ι?1??4B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?N@aÓ?!??<???;@)???S㥋?1?wJW3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvq?-??!?!??ע6@)9??v????1e?????2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????w?!??v?p? @)?????w?1??v?p? @:Preprocessing2U
Iterator::Model::ParallelMapV2"??u??q?!k;{8O?@)"??u??q?1k;{8O?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???3???!??fB?L@)??H?}m?1y`?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!c???@)Ǻ???f?1c???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??A?f??!E??ZT?=@)-C??6Z?1???M?V@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9g??????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+mq??,4@+mq??,4@!+mq??,4@      ??!       "	/?>:u???/?>:u???!/?>:u???*      ??!       2	??(\?¥???(\?¥?!??(\?¥?:	???!??????!???!???!???B      ??!       J	?Y.?????Y.????!?Y.????R      ??!       Z	?Y.?????Y.????!?Y.????JGPUYg??????b ?"E
)gradient_tape/sequential_3/dense_9/MatMulMatMul4 ??T??!4 ??T??"H
,gradient_tape/sequential_3/dense_10/MatMul_1MatMulAZ?????!?"=,T??"F
*gradient_tape/sequential_3/dense_10/MatMulMatMul???[1??!U?)???"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch??Cp̚?!????????"8
sequential_3/dense_10/MatMulMatMul3ۂ.?`??!L???-??"7
sequential_3/dense_9/MatMulMatMul?,,?љ?!??@N2h??"8
dense_10/kernel/Regularizer/SumSum??~'???!QZ03?~??"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam???_????!?,?*[??"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam???J`ݑ?!&?CTkK??"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam???J`ݑ?!Ev?XAi??Q      Y@Y[k???Z3@a)??RJ)T@q??W?J@yJCr7v??"?
both?Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?52.1625% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 