?	ɯb?wM@ɯb?wM@!ɯb?wM@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ɯb?wM@?F?q?IA@1???7@A"??u????Ik???u???*	43333??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle????????!=?B??gQ@)lxz?,C??1@.??2ZI@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?yX?5?;??!u?];?2@)yX?5?;??1u?];?2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache????QI??!?J??1?:@)=,Ԛ???1dwh?{?,@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???~j?t??!-??W(@)??~j?t??1-??W(@:Preprocessing2F
Iterator::Model????<,??!Ze?=	@)L7?A`???1??2Z?#@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch???_vOn?! s??R???)???_vOn?1 s??R???:Preprocessing2P
Iterator::Model::Prefetch-C??6j?!:N?Uf??)-C??6j?1:N?Uf??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?F?q?IA@?F?q?IA@!?F?q?IA@      ??!       "	???7@???7@!???7@*      ??!       2	"??u????"??u????!"??u????:	k???u???k???u???!k???u???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_202/Conv2D/Conv2DBackpropInputConv2DBackpropInput?o8?????!?o8?????"-
IteratorGetNext/_1_Send??????!g???]???"d
:gradient_tape/model/conv2d_202/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2?&w???!?KX	?h??"3
model/conv2d_202/Conv2DConv2D	?	??`??!?[m?u ??"d
:gradient_tape/model/conv2d_203/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterBN16????!_?3?ʳ??"3
model/conv2d_203/Conv2DConv2D??I??L??!?_]??"b
9gradient_tape/model/conv2d_203/Conv2D/Conv2DBackpropInputConv2DBackpropInput?:?F+??!xip.???"i
?gradient_tape/model/batch_normalization_96/FusedBatchNormGradV3FusedBatchNormGradV3???x#??!?~i$?C??"d
:gradient_tape/model/conv2d_201/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?gC?0"??!??0????"b
9gradient_tape/model/conv2d_201/Conv2D/Conv2DBackpropInputConv2DBackpropInputP?o\
#??!????????Q      Y@Yj?t??U@a??W[?:+@q??? ?<@y?Fؙ9??"?	
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?28.8052% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 