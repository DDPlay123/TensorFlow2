?	!?˛?C@!?˛?C@!!?˛?C@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-!?˛?C@? ?bG?4@1?$z?r1@A??????IC?Գ ???*	333333?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?Gx$(??!\=;nlN@)?sF????1W??;G@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??U??????!h???e@@)??A?f??1]t?E?5@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch???ܵ???!??J?,@)??ܵ???1??J?,@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???ZӼ???!pc?
%@)??ZӼ???1pc?
%@:Preprocessing2F
Iterator::Model(~??k	??!?|??B@)?lV}???19?????@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?I+?v?!?袋.???)?I+?v?1?袋.???:Preprocessing2P
Iterator::Model::PrefetchHP?s?r?!?|????)HP?s?r?1?|????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? ?bG?4@? ?bG?4@!? ?bG?4@      ??!       "	?$z?r1@?$z?r1@!?$z?r1@*      ??!       2	????????????!??????:	C?Գ ???C?Գ ???!C?Գ ???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_1_SendB??m?ż?!B??m?ż?"a
8gradient_tape/model/conv2d_47/Conv2D/Conv2DBackpropInputConv2DBackpropInput[??"^??!ΞAH???"5
model/re_lu_56/Relu_FusedConv2D???ϲ?!?p
g????"5
model/re_lu_55/Relu_FusedConv2D??a?FO??!????P??"c
9gradient_tape/model/conv2d_47/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.jlu???!??T%???"c
9gradient_tape/model/conv2d_48/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter
?G0????! hЬ#??"a
8gradient_tape/model/conv2d_48/Conv2D/Conv2DBackpropInputConv2DBackpropInput????׵??!9?:?gZ??"5
model/re_lu_57/Relu_FusedConv2D?B?????!???Jߍ??"c
9gradient_tape/model/conv2d_46/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}l?d[??!`?????"a
8gradient_tape/model/conv2d_46/Conv2D/Conv2DBackpropInputConv2DBackpropInputx?xS??!Ȃ<?X??Q      Y@Y??????U@a!"""""*@q)?:?E@y??^?ׇ?"?	
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?42.1328% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 