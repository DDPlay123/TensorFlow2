?	??*??A@??*??A@!??*??A@	??:??!????:??!??!??:??!??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??*??A@????0@1%???2@A46<???I?t_ά??Y?'?>???*	??????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???6?[??!\,'?qxP@)%??C???1Ä?M?_F@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?s??A??!秃??"5@)s??A??1秃??"5@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?X9??v??!F?Ma>@)??y?)??1~3۰1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?????ò?!f?4?a`)@)????ò?1f?4?a`)@:Preprocessing2F
Iterator::ModelΈ?????!???K??	@)X9??v???1??A??v@:Preprocessing2P
Iterator::Model::Prefetcha??+ei?!?/?2?+??)a??+ei?1?/?2?+??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?~j?t?h?!?	s???)?~j?t?h?1?	s???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??:??!??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????0@????0@!????0@      ??!       "	%???2@%???2@!%???2@*      ??!       2	46<???46<???!46<???:	?t_ά???t_ά??!?t_ά??B      ??!       J	?'?>????'?>???!?'?>???R      ??!       Z	?'?>????'?>???!?'?>???JGPUY??:??!??b ?"b
9gradient_tape/model/conv2d_127/Conv2D/Conv2DBackpropInputConv2DBackpropInput?3B????!?3B????"-
IteratorGetNext/_1_Send6re????!jZ??B??"d
:gradient_tape/model/conv2d_127/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'?8?!i??!?@?@h{??"6
model/re_lu_152/Relu_FusedConv2D?]??B??!?rL??"d
:gradient_tape/model/conv2d_128/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??u??β?!d???????"6
model/re_lu_153/Relu_FusedConv2D?ܜ׳??!*??I^???"b
9gradient_tape/model/conv2d_128/Conv2D/Conv2DBackpropInputConv2DBackpropInput2I?Զv??!P=?$5%??"d
:gradient_tape/model/conv2d_126/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)"??????!s ???"b
9gradient_tape/model/conv2d_126/Conv2D/Conv2DBackpropInputConv2DBackpropInputrc?*???!?UsK?H??"6
model/re_lu_151/Relu_FusedConv2D!Կꩩ??!?R????Q      Y@Y*?:??V@a?F)NB? @q??~?C?6@y??C?Ǎ?"?
both?Your program is POTENTIALLY input-bound because 45.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?22.6026% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 