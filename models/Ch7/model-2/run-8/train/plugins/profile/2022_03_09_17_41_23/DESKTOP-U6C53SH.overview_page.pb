?	?52;QA@?52;QA@!?52;QA@	z??1g??z??1g??!z??1g??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?52;QA@Н`?u?/@1??;???1@A\ A?c̝?I?H?"i7??Y?k?????*	effff?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???&S??!A????xP@)?q?????1????mI@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??e??a???!?Y?+?;@)??MbX??1s???.@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?,Ԛ????!1?9?a.@),Ԛ????11?9?a.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?	??g????!?@v?:H(@)	??g????1?@v?:H(@:Preprocessing2F
Iterator::ModelHP?sע?!?71??@)?|a2U??1?-Q ?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch{?G?zt?!崊????){?G?zt?1崊????:Preprocessing2P
Iterator::Model::Prefetchn??t?!26w+x??)n??t?126w+x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9z??1g??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Н`?u?/@Н`?u?/@!Н`?u?/@      ??!       "	??;???1@??;???1@!??;???1@*      ??!       2	\ A?c̝?\ A?c̝?!\ A?c̝?:	?H?"i7???H?"i7??!?H?"i7??B      ??!       J	?k??????k?????!?k?????R      ??!       Z	?k??????k?????!?k?????JGPUYz??1g??b ?"a
8gradient_tape/model/conv2d_67/Conv2D/Conv2DBackpropInputConv2DBackpropInputy???w???!y???w???"-
IteratorGetNext/_2_Recv\?7?9??!B??????"5
model/re_lu_80/Relu_FusedConv2D??B?Z??!??PaV??"c
9gradient_tape/model/conv2d_67/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltere?#?Vű?!?Y????"c
9gradient_tape/model/conv2d_68/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter+S???J??!???]???"5
model/re_lu_81/Relu_FusedConv2DEH??J/??!y?}@????"a
8gradient_tape/model/conv2d_68/Conv2D/Conv2DBackpropInputConv2DBackpropInput?C???+??!??9v???"c
9gradient_tape/model/conv2d_66/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Z?????!?_??:??"a
8gradient_tape/model/conv2d_66/Conv2D/Conv2DBackpropInputConv2DBackpropInput??m?[???!Z?fk?z??"5
model/re_lu_79/Relu_FusedConv2DǊ0????!?g~̳??Q      Y@Y??????U@a!"""""*@q?.\iA@yǾ??c??"?	
both?Your program is POTENTIALLY input-bound because 45.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?34.8231% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 