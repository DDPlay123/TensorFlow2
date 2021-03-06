?	?l;m??Q@?l;m??Q@!?l;m??Q@	????]??????]??!????]??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?l;m??Q@?8?	??G@1|???S?6@A%u???I1?*????Y????G??*	?????L?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??z?G???!??FX?)P@)s??A??1???d)G@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?!?lV}??!~T??U2@)!?lV}??1~T??U2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?m???{???!Tz??? >@)q=
ףp??1u?Y?1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?jM??St??!??Gu??(@)jM??St??1??Gu??(@:Preprocessing2F
Iterator::ModelV-???!?m۶m?@)tF??_??1+?4?rO@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?+e?Xw?!?%f-??)?+e?Xw?1?%f-??:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!??????)??_?Lu?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????]??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8?	??G@?8?	??G@!?8?	??G@      ??!       "	|???S?6@|???S?6@!|???S?6@*      ??!       2	%u???%u???!%u???:	1?*????1?*????!1?*????B      ??!       J	????G??????G??!????G??R      ??!       Z	????G??????G??!????G??JGPUY????]??b ?"a
8gradient_tape/model/conv2d_87/Conv2D/Conv2DBackpropInputConv2DBackpropInputd??????!d??????"-
IteratorGetNext/_1_Send[??5??!`Qמ?H??"2
model/conv2d_87/Conv2DConv2D???????!??;?%???"c
9gradient_tape/model/conv2d_87/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?N???!?~?uH??"2
model/conv2d_88/Conv2DConv2Dƻ?%&??!???Ƭ??"a
8gradient_tape/model/conv2d_88/Conv2D/Conv2DBackpropInputConv2DBackpropInputa???3???!Jo?7???"c
9gradient_tape/model/conv2d_88/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterHNw?{Ǫ?!Y??|d??"i
?gradient_tape/model/batch_normalization_12/FusedBatchNormGradV3FusedBatchNormGradV3Q?3j??!?????x??"c
9gradient_tape/model/conv2d_86/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter̦?"??!\??l	???"a
8gradient_tape/model/conv2d_86/Conv2D/Conv2DBackpropInputConv2DBackpropInput޶?????!ʓ??9???Q      Y@YF?лS@a???o?5@q??ڣbL@y??3$&d??"?	
both?Your program is POTENTIALLY input-bound because 66.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?56.7706% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 