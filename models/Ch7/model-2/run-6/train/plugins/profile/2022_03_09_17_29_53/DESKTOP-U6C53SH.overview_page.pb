?	?HM???B@?HM???B@!?HM???B@	? ????? ????!? ????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?HM???B@1???61@1g????3@A?X?? ??I:?S?????Y?>:u峌?*	??????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??ZӼ???!G x??J@)??4?8E??1˂?RthD@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?	?c???!?Y?-~D@)}??b???1???u?#7@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??c]?F??!?,??F?0@)?c]?F??1?,??F?0@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??z6?>??!???/??(@)?z6?>??1???/??(@:Preprocessing2F
Iterator::ModelM??St$??!??4?@){?G?z??1???d?@:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!????|???)??_?Lu?1????|???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatcha2U0*?s?!?d??K???)a2U0*?s?1?d??K???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9? ????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1???61@1???61@!1???61@      ??!       "	g????3@g????3@!g????3@*      ??!       2	?X?? ???X?? ??!?X?? ??:	:?S?????:?S?????!:?S?????B      ??!       J	?>:u峌??>:u峌?!?>:u峌?R      ??!       Z	?>:u峌??>:u峌?!?>:u峌?JGPUY? ????b ?"-
IteratorGetNext/_1_SendZs???f??!Zs???f??"a
8gradient_tape/model/conv2d_57/Conv2D/Conv2DBackpropInputConv2DBackpropInput????I??!???#i???"5
model/re_lu_68/Relu_FusedConv2D???????!,??A?S??"5
model/re_lu_67/Relu_FusedConv2D?==&?y??!?k?2??"C
%gradient_tape/model/re_lu_66/ReluGradReluGrad	?r????!?9o1??"c
9gradient_tape/model/conv2d_57/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterϐXt????!3 7(0??"5
model/re_lu_69/Relu_FusedConv2D>~MW?j??!?g??????"c
9gradient_tape/model/conv2d_58/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb??3??!???????"a
8gradient_tape/model/conv2d_58/Conv2D/Conv2DBackpropInputConv2DBackpropInput??0???!;Q?Ap??"Y
8gradient_tape/model/max_pooling2d_11/MaxPool/MaxPoolGradMaxPoolGrad?hvqkJ??!?h?????Q      Y@Y??????U@a!"""""*@q?ɐA?&D@yz???,??"?	
both?Your program is POTENTIALLY input-bound because 45.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?40.3036% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 