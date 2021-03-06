?	?k
dv?A@?k
dv?A@!?k
dv?A@	?S??q???S??q??!?S??q??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?k
dv?A@
-????0@1F?Sw?1@Aj?t???I?蹅????Y?vN?@???*	?????E?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?????K7??!?\yc?P@)?HP???1?@?W?H@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?.?!??u??!L???n?2@).?!??u??1L???n?2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?T㥛? ??!?o?C,?;@)p_?Q??1???F4.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?8gDio???!T??.)@)8gDio???1T??.)@:Preprocessing2F
Iterator::Model?Zd;??!?y??7?@)?&1???1Y??,?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??_?Lu?!k??Kr??)??_?Lu?1k??Kr??:Preprocessing2P
Iterator::Model::Prefetch{?G?zt?!"???U???){?G?zt?1"???U???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?S??q??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
-????0@
-????0@!
-????0@      ??!       "	F?Sw?1@F?Sw?1@!F?Sw?1@*      ??!       2	j?t???j?t???!j?t???:	?蹅?????蹅????!?蹅????B      ??!       J	?vN?@????vN?@???!?vN?@???R      ??!       Z	?vN?@????vN?@???!?vN?@???JGPUY?S??q??b ?"-
IteratorGetNext/_1_Send??L?z???!??L?z???"a
8gradient_tape/model/conv2d_62/Conv2D/Conv2DBackpropInputConv2DBackpropInput	?Js??!??Kb???"c
9gradient_tape/model/conv2d_60/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`6?lC|??!?e?0???"5
model/re_lu_74/Relu_FusedConv2Dओ.N???!<?
????"c
9gradient_tape/model/conv2d_62/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter֨PO? ??!r?ޫ?'??"a
8gradient_tape/model/conv2d_63/Conv2D/Conv2DBackpropInputConv2DBackpropInput?)B?η??!?????J??"5
model/re_lu_75/Relu_FusedConv2D?N?k???!?Ϸ,????"c
9gradient_tape/model/conv2d_63/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??7???!?~?????"c
9gradient_tape/model/conv2d_61/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter褗Se]??!R<?B?<??"a
8gradient_tape/model/conv2d_61/Conv2D/Conv2DBackpropInputConv2DBackpropInputN)?p??!???????Q      Y@Y??????U@a!"""""*@q?N??lG@y??Sl9???"?
both?Your program is POTENTIALLY input-bound because 46.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?46.0111% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 