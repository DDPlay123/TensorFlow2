?	!??^?D@!??^?D@!!??^?D@	$[?T??r?$[?T??r?!$[?T??r?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6!??^?D@v??^3@1?V횐?3@A+??????I{L?4???Y$Di?]?*	???????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??~j?t???!?G?RަO@)???K7???1?<???F@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?)\???(??!?^??D"2@))\???(??1?^??D"2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?"?uq??!?|@???>@)-????ƻ?1?E????1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?jM??S??!?l?X?-*@)jM??S??1?l?X?-*@:Preprocessing2F
Iterator::ModelO??e?c??!?FJ??@)M??St$??1t?|3?@:Preprocessing2P
Iterator::Model::Prefetch?HP?x?!?G???)?HP?x?1?G???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?~j?t?x?!?G?Rަ??)?~j?t?x?1?G?Rަ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9$[?T??r?>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v??^3@v??^3@!v??^3@      ??!       "	?V횐?3@?V횐?3@!?V횐?3@*      ??!       2	+??????+??????!+??????:	{L?4???{L?4???!{L?4???B      ??!       J	$Di?]?$Di?]?!$Di?]?R      ??!       Z	$Di?]?$Di?]?!$Di?]?JGPUY$[?T??r?b ?"5
model/re_lu_45/Relu_FusedConv2D{t?????!{t?????"5
model/re_lu_44/Relu_FusedConv2D???ծ??!?T*?N???"-
IteratorGetNext/_1_Sendip"4:w??!Ɲ??U??"a
8gradient_tape/model/conv2d_37/Conv2D/Conv2DBackpropInputConv2DBackpropInput#?Y??p??!,46 r??"c
9gradient_tape/model/conv2d_37/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>??)o???!?a?78??"a
8gradient_tape/model/conv2d_38/Conv2D/Conv2DBackpropInputConv2DBackpropInput!?U?Z!??!????L*??"c
9gradient_tape/model/conv2d_38/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterL3????!{??<<??"c
9gradient_tape/model/conv2d_35/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?"????!??~???"c
9gradient_tape/model/conv2d_36/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??=I??!???Y?Z??"a
8gradient_tape/model/conv2d_36/Conv2D/Conv2DBackpropInputConv2DBackpropInputs?eB\???!gC?Fy??Q      Y@Y??????U@a!"""""*@q?X?,%0@@y7?_??֏?"?	
both?Your program is POTENTIALLY input-bound because 47.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?32.3761% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 