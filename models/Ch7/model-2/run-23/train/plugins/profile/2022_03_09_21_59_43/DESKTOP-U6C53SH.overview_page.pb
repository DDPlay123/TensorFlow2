?	??zO??A@??zO??A@!??zO??A@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??zO??A@?8???/@1S?Q?2@AΈ?????IޫV&?R??*	??????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???D????!>/e%??P@)?J?4??1}?u??IG@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?OjM???!???`?4@)OjM???1???`?4@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??v??/??! ????<@)?,C????1????.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???9#J{??!?<?_*@)??9#J{??1?<?_*@:Preprocessing2F
Iterator::Model??d?`T??!???3P?@)9??v????10?mqi@:Preprocessing2P
Iterator::Model::Prefetchn??t?!=??	?+??)n??t?1=??	?+??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch????Mbp?!ٸK?.??)????Mbp?1ٸK?.??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8???/@?8???/@!?8???/@      ??!       "	S?Q?2@S?Q?2@!S?Q?2@*      ??!       2	Έ?????Έ?????!Έ?????:	ޫV&?R??ޫV&?R??!ޫV&?R??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_142/Conv2D/Conv2DBackpropInputConv2DBackpropInput??g$Ӹ??!??g$Ӹ??"-
IteratorGetNext/_2_RecvS??Д???!????????"d
:gradient_tape/model/conv2d_142/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? ???4??!???_?+??"6
model/re_lu_170/Relu_FusedConv2D???2 ??!O??????"d
:gradient_tape/model/conv2d_143/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD?夌???!??F	???"6
model/re_lu_171/Relu_FusedConv2D?2??`??!I?o????"b
9gradient_tape/model/conv2d_143/Conv2D/Conv2DBackpropInputConv2DBackpropInputɑ?l?B??!??me???"d
:gradient_tape/model/conv2d_141/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??1??!??t??"b
9gradient_tape/model/conv2d_141/Conv2D/Conv2DBackpropInputConv2DBackpropInput??7b?O??!2C?z???"6
model/re_lu_169/Relu_FusedConv2D?*x???!+?j?zS??Q      Y@Y#?n?H&W@a?m?t?@qys+???6@yd?G?⸌?"?	
both?Your program is POTENTIALLY input-bound because 45.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?22.8238% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 