?	b????4@b????4@!b????4@	$x8db@??$x8db@??!$x8db@??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-b????4@???T???1=dʇ?R1@I?E??G??Y?mO???n?*	fffff?d@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle? ?o_???!q`?6_?T@) ?o_???1q`?6_?T@:Preprocessing2F
Iterator::Modelw-!?l??!2??UF*@)K?=?U??1>$\8["@:Preprocessing2P
Iterator::Model::PrefetchF%u?{?!???Su?@)F%u?{?1???Su?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?+e?Xw?!?0k?Z@)?+e?Xw?1?0k?Z@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9$x8db@??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???T??????T???!???T???      ??!       "	=dʇ?R1@=dʇ?R1@!=dʇ?R1@*      ??!       2      ??!       :	?E??G???E??G??!?E??G??B      ??!       J	?mO???n??mO???n?!?mO???n?R      ??!       Z	?mO???n??mO???n?!?mO???n?JGPUY$x8db@??b ?"-
IteratorGetNext/_1_Send ???! ???"j
Mgradient_tape/model-1/conv2d_4/Conv2D/Conv2DBackpropInput:Conv2DBackpropInputUnknown-?SWל??!??5-?R??"l
Ogradient_tape/model-1/conv2d_5/Conv2D/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown.???"???!ڒRG????"?
"model-1/conv2d_4/Relu:_FusedConv2DUnknown???D???!6?H0???"l
Ogradient_tape/model-1/conv2d_4/Conv2D/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown????D??!????Y8??"?
"model-1/conv2d_5/Relu:_FusedConv2DUnknown3??????!? ?_?R??"j
Mgradient_tape/model-1/conv2d_5/Conv2D/Conv2DBackpropInput:Conv2DBackpropInputUnknown?v&?w??!{?C?????"l
Ogradient_tape/model-1/conv2d_3/Conv2D/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown-????\??!Bܽ??"j
Mgradient_tape/model-1/conv2d_3/Conv2D/Conv2DBackpropInput:Conv2DBackpropInputUnknown/?WS?>??!A?wѬK??"?
"model-1/conv2d_3/Relu:_FusedConv2DUnknown?A?|=??!`=G?????Q      Y@Y??????W@a??????@qw?߃?n'@y??4?q6??"?
both?Your program is POTENTIALLY input-bound because 7.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?11.716% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 