?	_y??"eL@_y??"eL@!_y??"eL@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-_y??"eL@whX??@@1C???7@AR*?	????I?O??????*	     ??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??*??	??!?o????O@)???1????1?q?qH@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??ͪ??V??!??O????@)?? ???1??D?o-3@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?$(~??k??!ΡbAs?,@)$(~??k??1ΡbAs?,@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?-??臨?!*4??)@)-??臨?1*4??)@:Preprocessing2F
Iterator::Model?]K?=??!
????@)ˡE?????10?L?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchlxz?,C|?!&rk????)lxz?,C|?1&rk????:Preprocessing2P
Iterator::Model::Prefetch?HP?x?!h?p??*??)?HP?x?1h?p??*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	whX??@@whX??@@!whX??@@      ??!       "	C???7@C???7@!C???7@*      ??!       2	R*?	????R*?	????!R*?	????:	?O???????O??????!?O??????B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_1_Send?/?R ܵ?!?/?R ܵ?"a
8gradient_tape/model/conv2d_77/Conv2D/Conv2DBackpropInputConv2DBackpropInputx??z?ϴ?!???f?U??"c
9gradient_tape/model/conv2d_77/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?&?Ei˱?!?'?	?;??"2
model/conv2d_77/Conv2DConv2D??]/h???!*G?
^???"P
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3q-?^????!،?6y??"2
model/conv2d_78/Conv2DConv2D?b0??Ū?!6??I*g??"a
8gradient_tape/model/conv2d_78/Conv2D/Conv2DBackpropInputConv2DBackpropInput??? ?n??!2?-	???"c
9gradient_tape/model/conv2d_78/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterp??sh??!??)???"f
<gradient_tape/model/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3AJ?)??!??寝C??"c
9gradient_tape/model/conv2d_76/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterғY?6&??!?b????Q      Y@YF?лS@a???o?5@q???w??E@yk??J??"?	
both?Your program is POTENTIALLY input-bound because 56.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?43.9653% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 