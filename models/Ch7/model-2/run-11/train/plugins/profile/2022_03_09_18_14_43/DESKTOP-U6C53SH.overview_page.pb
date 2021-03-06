?	???>9?O@???>9?O@!???>9?O@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???>9?O@6?;N??B@1??)?d8@A???B?i??I?˙?
???*	33333߄@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?{?/L?
??!??9??Q@)?C?l????1N{i?<	E@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??H?}??!C9f35<@)?H?}??1C9f35<@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?V-???!?????`9@)?^)???1{<????*@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?jM??S??!?????'@)jM??S??1?????'@:Preprocessing2F
Iterator::Model?HP???!??e??9@)jM????1??d??@:Preprocessing2P
Iterator::Model::Prefetch?g??s?u?!??6&?d??)?g??s?u?1??6&?d??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchHP?s?r?!???

??)HP?s?r?1???

??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6?;N??B@6?;N??B@!6?;N??B@      ??!       "	??)?d8@??)?d8@!??)?d8@*      ??!       2	???B?i?????B?i??!???B?i??:	?˙?
????˙?
???!?˙?
???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"a
8gradient_tape/model/conv2d_83/Conv2D/Conv2DBackpropInputConv2DBackpropInput???
?a??!???
?a??"c
9gradient_tape/model/conv2d_83/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb??$???!??ӏ^???"-
IteratorGetNext/_2_Recv?w?N???!????B???"a
8gradient_tape/model/conv2d_82/Conv2D/Conv2DBackpropInputConv2DBackpropInput???Y?8??!?FI????"2
model/conv2d_82/Conv2DConv2D???K~???!JJ?Q??"c
9gradient_tape/model/conv2d_82/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?@?<^???!a?U?j??"2
model/conv2d_83/Conv2DConv2D???G???!???A Q??"h
>gradient_tape/model/batch_normalization_6/FusedBatchNormGradV3FusedBatchNormGradV3|a??[??!Ү?چ??"c
9gradient_tape/model/conv2d_81/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?T?????!!T8]????"a
8gradient_tape/model/conv2d_81/Conv2D/Conv2DBackpropInputConv2DBackpropInput?cV?ʝ?!>KY???Q      Y@YF?лS@a???o?5@q?-???N@ym"e4???"?	
both?Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?61.3961% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 