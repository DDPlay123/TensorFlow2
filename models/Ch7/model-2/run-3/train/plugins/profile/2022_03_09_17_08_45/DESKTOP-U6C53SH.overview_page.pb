?	'N?w(?B@'N?w(?B@!'N?w(?B@	C??U:??C??U:??!C??U:??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6'N?w(?B@?5=((?1@1?_#I?1@A????岡?Iѯ???3??Y??u6????*	????̬?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?:#J{?/??!P/?7?P@)?c]?F??1?.?ѡG@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?c?=yX??!?_?;M3@)c?=yX??1?_?;M3@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??(??0??!???s??=@)?"??~j??1_??
?0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?I.?!????!??\<??)@)I.?!????1??\<??)@:Preprocessing2F
Iterator::Model??d?`T??!2 H???@)?
F%u??1QhӢ???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch9??v??z?!+?^p??)9??v??z?1+?^p??:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!"?宱&??)??_?Lu?1"?宱&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9C??U:??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?5=((?1@?5=((?1@!?5=((?1@      ??!       "	?_#I?1@?_#I?1@!?_#I?1@*      ??!       2	????岡?????岡?!????岡?:	ѯ???3??ѯ???3??!ѯ???3??B      ??!       J	??u6??????u6????!??u6????R      ??!       Z	??u6??????u6????!??u6????JGPUYC??U:??b ?"-
IteratorGetNext/_1_Send??PHޅ??!??PHޅ??"a
8gradient_tape/model/conv2d_42/Conv2D/Conv2DBackpropInputConv2DBackpropInput?ik?y??!h;?Y???"5
model/re_lu_50/Relu_FusedConv2D??M??۲?!?B'????"c
9gradient_tape/model/conv2d_42/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??k2	??!?J=?y??"5
model/re_lu_51/Relu_FusedConv2D???!O???!⿸????"a
8gradient_tape/model/conv2d_43/Conv2D/Conv2DBackpropInputConv2DBackpropInput?^"?t???!Ϋ?X?#??"c
9gradient_tape/model/conv2d_43/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?/??p??!??Ɩ?Q??"c
9gradient_tape/model/conv2d_41/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter՛?v??!??fis??"X
7gradient_tape/model/max_pooling2d_8/MaxPool/MaxPoolGradMaxPoolGrad/?ƁXæ?!????????"a
8gradient_tape/model/conv2d_41/Conv2D/Conv2DBackpropInputConv2DBackpropInputC?UGMM??!?H[?s$??Q      Y@Y??????U@a!"""""*@q???4?B@y!?$}w???"?	
both?Your program is POTENTIALLY input-bound because 48.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?37.0954% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 