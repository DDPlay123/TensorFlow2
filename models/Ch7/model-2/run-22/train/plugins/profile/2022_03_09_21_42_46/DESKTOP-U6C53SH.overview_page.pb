?	?+I???A@?+I???A@!?+I???A@	??????????????!???????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?+I???A@?<?+Ja0@1??m3
2@AˡE?????It\??JK??Y??25	ް?*	????̀?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?v??????!?^?_?P@)$(~??k??1???Ri?F@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??p=
ף??!?<??5@)?p=
ף??1?<??5@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?o??ʡ??!l???ߊ<@)_)?Ǻ??1????P0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??:pΈ??!?Auh?t(@)?:pΈ??1?Auh?t(@:Preprocessing2F
Iterator::Modelvq?-??!??;?Y@) ?o_Ή?1?k?.@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??ZӼ?t?!3??????)??ZӼ?t?13??????:Preprocessing2P
Iterator::Model::Prefetch-C??6j?!??8\K??)-C??6j?1??8\K??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?<?+Ja0@?<?+Ja0@!?<?+Ja0@      ??!       "	??m3
2@??m3
2@!??m3
2@*      ??!       2	ˡE?????ˡE?????!ˡE?????:	t\??JK??t\??JK??!t\??JK??B      ??!       J	??25	ް???25	ް?!??25	ް?R      ??!       Z	??25	ް???25	ް?!??25	ް?JGPUY???????b ?"b
9gradient_tape/model/conv2d_137/Conv2D/Conv2DBackpropInputConv2DBackpropInput?E????!?E????"-
IteratorGetNext/_1_Send?bkzi???!??X7????"d
:gradient_tape/model/conv2d_137/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterm???{??!???߾[??"6
model/re_lu_164/Relu_FusedConv2DR?\+g??!p???5??"6
model/re_lu_165/Relu_FusedConv2Dd!?????!Ɍ"{???"d
:gradient_tape/model/conv2d_138/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterl?K??ϲ?!???;???"b
9gradient_tape/model/conv2d_138/Conv2D/Conv2DBackpropInputConv2DBackpropInput?z?????!E5??(??"d
:gradient_tape/model/conv2d_136/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb?1	ᾩ?![R??????"b
9gradient_tape/model/conv2d_136/Conv2D/Conv2DBackpropInputConv2DBackpropInput?X??_y??!??K??"6
model/re_lu_163/Relu_FusedConv2D?˭?ۿ??!??id????Q      Y@YQ?"?W@a??׽?u@q???FI?4@yWc zCm??"?	
both?Your program is POTENTIALLY input-bound because 46.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?20.5402% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 