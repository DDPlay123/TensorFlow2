?	?vR~?P@?vR~?P@!?vR~?P@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?vR~?P@?,z??E@1??0???5@A?:pΈҞ?I?V?????*	???????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???ܵ???!Wh'#\I@)?????K??1V?A?WC@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??	?c??!/?JйF@)?W?2??1??.?mA@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch???9#J{??!Ę?(@)??9#J{??1Ę?(@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl????x?&??!MiXI?2%@)???x?&??1MiXI?2%@:Preprocessing2F
Iterator::Model??ׁsF??!CV?/	@)?<,Ԛ???1???dz@:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!????/S??)??_?Lu?1????/S??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch/n??r?!??~?cF??)/n??r?1??~?cF??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 64.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?,z??E@?,z??E@!?,z??E@      ??!       "	??0???5@??0???5@!??0???5@*      ??!       2	?:pΈҞ??:pΈҞ?!?:pΈҞ?:	?V??????V?????!?V?????B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_2_Recv??;5? ??!??;5? ??"a
8gradient_tape/model/conv2d_97/Conv2D/Conv2DBackpropInputConv2DBackpropInputg?Yzy͵?!???Wg??"2
model/conv2d_97/Conv2DConv2DH@5???!??%????"c
9gradient_tape/model/conv2d_97/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?-????!??-X*???"2
model/conv2d_98/Conv2DConv2D??ϭ???!j???^??"c
9gradient_tape/model/conv2d_98/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterV?:IJ???!?}??"a
8gradient_tape/model/conv2d_98/Conv2D/Conv2DBackpropInputConv2DBackpropInputTB??^??!?g?R????"i
?gradient_tape/model/batch_normalization_24/FusedBatchNormGradV3FusedBatchNormGradV3KՎe?\??!?B??y???"c
9gradient_tape/model/conv2d_96/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??1a???!???"??"a
8gradient_tape/model/conv2d_96/Conv2D/Conv2DBackpropInputConv2DBackpropInput8??U?̠?!$?L??%??Q      Y@YF?лS@a???o?5@q1G???mO@y??J?Dj??"?	
both?Your program is POTENTIALLY input-bound because 64.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?62.8589% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 