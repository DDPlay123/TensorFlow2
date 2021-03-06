?	Pr?Md?5@Pr?Md?5@!Pr?Md?5@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Pr?Md?5@?%?<??1??0B`3@AP ?Ȓ9??IV??f???*	33333Cd@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?D?l?????!??}HU@)D?l?????1??}HU@:Preprocessing2F
Iterator::Model???{????!??Im$@)A??ǘ???1oL???c@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?q?????!???,T?@)?q?????1???,T?@:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!?"?ũ	@)??_?Lu?1?"?ũ	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?%?<???%?<??!?%?<??      ??!       "	??0B`3@??0B`3@!??0B`3@*      ??!       2	P ?Ȓ9??P ?Ȓ9??!P ?Ȓ9??:	V??f???V??f???!V??f???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_2_Recv??]???!??]???"j
Mgradient_tape/model-1/conv2d_2/Conv2D/Conv2DBackpropInput:Conv2DBackpropInputUnknown?{Ti???!????AO??"l
Ogradient_tape/model-1/conv2d_3/Conv2D/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown??bH6??!???$u??"j
Mgradient_tape/model-1/conv2d_1/Conv2D/Conv2DBackpropInput:Conv2DBackpropInputUnknowno???	??!m??͞7??"K
.gradient_tape/model-1/conv2d/ReluGrad:ReluGradUnknown?MΓ???!?B
????"?
"model-1/conv2d_3/Relu:_FusedConv2DUnknown??@??T??!?|??A7??"?
"model-1/conv2d_2/Relu:_FusedConv2DUnknown/?bS???!??9^????"l
Ogradient_tape/model-1/conv2d_2/Conv2D/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownҮ??@??!?q??????"j
Mgradient_tape/model-1/conv2d_3/Conv2D/Conv2DBackpropInput:Conv2DBackpropInputUnknown?!????!??`5???"j
Mgradient_tape/model-1/conv2d/Conv2D/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownS?`"?ާ?!3??9??Q      Y@Y???!??W@a????p@q??B???#@y'?QB???"?

both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 