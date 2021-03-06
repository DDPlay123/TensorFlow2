?	^??I??N@^??I??N@!^??I??N@	??cT????cT??!??cT??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6^??I??N@'jin??B@1
?5?Y6@A???_vO??I???'??Y?p?;??*	???????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?????????!<n???[N@)??d?`T??1????'pF@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?)\???(??!??d?h<A@)?5^?I??1??@?6@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?R'??????!???Ϯ/@)R'??????1???Ϯ/@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??MbX9??!d??;??(@)?MbX9??1d??;??(@:Preprocessing2F
Iterator::Modelc?ZB>???!???ն@)?N@aÓ?1?Bk1@:Preprocessing2P
Iterator::Model::Prefetch?~j?t?x?!?$2???)?~j?t?x?1?$2???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??_vOv?!?} ?~??)??_vOv?1?} ?~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??cT??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'jin??B@'jin??B@!'jin??B@      ??!       "	
?5?Y6@
?5?Y6@!
?5?Y6@*      ??!       2	???_vO?????_vO??!???_vO??:	???'?????'??!???'??B      ??!       J	?p?;???p?;??!?p?;??R      ??!       Z	?p?;???p?;??!?p?;??JGPUY??cT??b ?"-
IteratorGetNext/_1_Send??Vh????!??Vh????"b
9gradient_tape/model/conv2d_107/Conv2D/Conv2DBackpropInputConv2DBackpropInput??.?????!̣B[??"3
model/conv2d_107/Conv2DConv2D?+,E????!??????"d
:gradient_tape/model/conv2d_107/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????f??!?N?YL??"d
:gradient_tape/model/conv2d_108/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?4??!7KI????"3
model/conv2d_108/Conv2DConv2D?L?\???!?t?4p6??"b
9gradient_tape/model/conv2d_108/Conv2D/Conv2DBackpropInputConv2DBackpropInputB@s?
J??!?? ?????"i
?gradient_tape/model/batch_normalization_36/FusedBatchNormGradV3FusedBatchNormGradV3?_?&?ڤ?!ӈ?
;??"d
:gradient_tape/model/conv2d_106/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!S?>????"b
9gradient_tape/model/conv2d_106/Conv2D/Conv2DBackpropInputConv2DBackpropInputf???????!??H<:???Q      Y@YF?лS@a???o?5@q??u?F@yR\?եĂ?"?	
both?Your program is POTENTIALLY input-bound because 61.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?45.8317% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 