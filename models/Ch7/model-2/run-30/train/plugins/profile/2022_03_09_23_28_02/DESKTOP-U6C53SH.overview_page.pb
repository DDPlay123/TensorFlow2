?	χg	2?L@χg	2?L@!χg	2?L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-χg	2?L@J^?c@R@@1??я?C7@A?!??u???IQ?O?I???*	     0?@2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?K?46??!o?FAoH@)??&S??1q>?c?C@:Preprocessing2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle????<,???!<??s?<G@)?W[?????1??B?/d@@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?V-????!c1#3b+@)V-????1c1#3b+@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?4??7?´?!?W?u?!@)4??7?´?1?W?u?!@:Preprocessing2F
Iterator::Model??ʡE???!?+Q?@)?? ?rh??1(?{??'@:Preprocessing2P
Iterator::Model::Prefetch;?O??nr?!?_??????);?O??nr?1?_??????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchy?&1?l?!??V?m???)y?&1?l?1??V?m???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J^?c@R@@J^?c@R@@!J^?c@R@@      ??!       "	??я?C7@??я?C7@!??я?C7@*      ??!       2	?!??u????!??u???!?!??u???:	Q?O?I???Q?O?I???!Q?O?I???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_177/Conv2D/Conv2DBackpropInputConv2DBackpropInput???	$??!???	$??"-
IteratorGetNext/_2_Recv  ?`???!?[?"?^??"d
:gradient_tape/model/conv2d_177/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterY?M??#??!??\C????"3
model/conv2d_177/Conv2DConv2D??e?????!?ے???"d
:gradient_tape/model/conv2d_178/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.S?.t???!Y?:J??"3
model/conv2d_178/Conv2DConv2D ?XO????!}"?"???"b
9gradient_tape/model/conv2d_178/Conv2D/Conv2DBackpropInputConv2DBackpropInputj?2??h??!J|~!.m??"d
:gradient_tape/model/conv2d_176/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???/N/??!N???	??"i
?gradient_tape/model/batch_normalization_66/FusedBatchNormGradV3FusedBatchNormGradV3?@?????!\??O?G??"b
9gradient_tape/model/conv2d_176/Conv2D/Conv2DBackpropInputConv2DBackpropInput?dS ????!???A ???Q      Y@YW;???U@a#G%޸?)@qAG??c=@y:7Z???"?	
both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?29.3897% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 