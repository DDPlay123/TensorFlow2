?	?_{f?J@?_{f?J@!?_{f?J@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?_{f?J@s?4?F=@1???l?F7@A?!??u???I??V|C???*?????-?@)      @=2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??+e?X??!Xؤf?P@)???9#J??1???/?H@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??3??7???!ǁ?_h<@)$(~??k??1???^?Y2@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??H?}??!???z?0@)?H?}??1???z?0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??HP???!8
???$@)?HP???18
???$@:Preprocessing2F
Iterator::ModelvOjM??!o?e[?T@)D?l?????1?????@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchy?&1?l?!Y?]?m3??)y?&1?l?1Y?]?m3??:Preprocessing2P
Iterator::Model::Prefetch-C??6j?!??滜??)-C??6j?1??滜??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s?4?F=@s?4?F=@!s?4?F=@      ??!       "	???l?F7@???l?F7@!???l?F7@*      ??!       2	?!??u????!??u???!?!??u???:	??V|C?????V|C???!??V|C???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_167/Conv2D/Conv2DBackpropInputConv2DBackpropInput????ZZ??!????ZZ??"-
IteratorGetNext/_2_Recvt?J???!8?ŋ?|??"d
:gradient_tape/model/conv2d_167/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?V?	گ?!???8s??"3
model/conv2d_167/Conv2DConv2Dj??7_??!&+PM(???"d
:gradient_tape/model/conv2d_168/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?.??=??!??U
????"3
model/conv2d_168/Conv2DConv2D???%????!??
/w@??"b
9gradient_tape/model/conv2d_168/Conv2D/Conv2DBackpropInputConv2DBackpropInput˖?]???!Z}?????"d
:gradient_tape/model/conv2d_166/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterPt?a????!LDW]?:??"i
?gradient_tape/model/batch_normalization_54/FusedBatchNormGradV3FusedBatchNormGradV3??????!&3l{??"b
9gradient_tape/model/conv2d_166/Conv2D/Conv2DBackpropInputConv2DBackpropInput??w>???!77?Sζ??Q      Y@Y0??>??U@a}?	?[+@q????q??@y?Ȝ??'??"?	
both?Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?31.826% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 