?	?L?T?A@?L?T?A@!?L?T?A@	?-'??at??-'??at?!?-'??at?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?L?T?A@?1Y??0@1?)??F?1@A)\???(??IaE|???YK?8???\?*	?????u?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??6?[ ??!???F??P@)??6???1?Hd?J@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?-!?lV??!(??f?8<@)?????K??1P?]K??.@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?/?$???!???l?p,@)/?$???1???l?p,@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???N@a??!?}?o?)@)??N@a??1?}?o?)@:Preprocessing2F
Iterator::Model??q????!@?$?67@)??_vO??17z???@@:Preprocessing2P
Iterator::Model::Prefetch?g??s?u?!#??1???)?g??s?u?1#??1???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch;?O??nr?!??8?`??);?O??nr?1??8?`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?-'??at?>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1Y??0@?1Y??0@!?1Y??0@      ??!       "	?)??F?1@?)??F?1@!?)??F?1@*      ??!       2	)\???(??)\???(??!)\???(??:	aE|???aE|???!aE|???B      ??!       J	K?8???\?K?8???\?!K?8???\?R      ??!       Z	K?8???\?K?8???\?!K?8???\?JGPUY?-'??at?b ?"a
8gradient_tape/model/conv2d_72/Conv2D/Conv2DBackpropInputConv2DBackpropInputB#??nD??!B#??nD??"-
IteratorGetNext/_1_Send?Xq#????!???rX???"5
model/re_lu_86/Relu_FusedConv2D???r?ղ?!(?x?????"c
9gradient_tape/model/conv2d_72/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?6??V??!晾
e??"5
model/re_lu_87/Relu_FusedConv2D??!???!?GL????"a
8gradient_tape/model/conv2d_73/Conv2D/Conv2DBackpropInputConv2DBackpropInput? 4?h???!? 
?av??"c
9gradient_tape/model/conv2d_73/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ???⇱?!G6%^???"c
9gradient_tape/model/conv2d_71/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??s???!f2l?2??"a
8gradient_tape/model/conv2d_71/Conv2D/Conv2DBackpropInputConv2DBackpropInput?61?K??!??E?yw??"5
model/re_lu_85/Relu_FusedConv2D?JfG2??!v8?????Q      Y@Y??????U@a!"""""*@q???LtD@y3?&ò??"?	
both?Your program is POTENTIALLY input-bound because 47.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?40.9067% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 