?	???8+?P@???8+?P@!???8+?P@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???8+?P@*7QKs?D@1Թ??7@A#K?X??I>?(??*	???????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???ͪ????!X?<-P@)?镲q??1?[??"?H@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?=
ףp=??!??[?â?@)? ?	???1ݚ)3@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch????????!VUUUU?.@)???????1VUUUU?.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???D????!Ix?5?)@)??D????1Ix?5?)@:Preprocessing2F
Iterator::Model?ݓ??Z??!UUUUUU@)_?Qڋ?1O???E? @:Preprocessing2P
Iterator::Model::Prefetch?g??s?u?!x?5?,??)?g??s?u?1x?5?,??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatcha2U0*?s?!C{	?%???)a2U0*?s?1C{	?%???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*7QKs?D@*7QKs?D@!*7QKs?D@      ??!       "	Թ??7@Թ??7@!Թ??7@*      ??!       2	#K?X??#K?X??!#K?X??:	>?(??>?(??!>?(??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_2_Recv??(?Vʵ?!??(?Vʵ?"a
8gradient_tape/model/conv2d_92/Conv2D/Conv2DBackpropInputConv2DBackpropInput`?????!???N~]??"2
model/conv2d_92/Conv2DConv2D?.?.td??!?0?Z?v??"c
9gradient_tape/model/conv2d_90/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr??G?M??!??=?????"c
9gradient_tape/model/conv2d_92/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???^en??!i??2??"2
model/conv2d_93/Conv2DConv2Db?41?˪?!4?:?>???"c
9gradient_tape/model/conv2d_93/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltero??#?m??!bӹ?????"a
8gradient_tape/model/conv2d_93/Conv2D/Conv2DBackpropInputConv2DBackpropInputP$x?*\??!?׈t%??"i
?gradient_tape/model/batch_normalization_18/FusedBatchNormGradV3FusedBatchNormGradV3"?@????!P?????"c
9gradient_tape/model/conv2d_91/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(?P?????!?)?????Q      Y@YF?лS@a???o?5@qu;???pM@y?????"?	
both?Your program is POTENTIALLY input-bound because 62.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?58.8805% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 