?	??ݯFB@??ݯFB@!??ݯFB@	T??5o???T??5o???!T??5o???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??ݯFB@?????3@1Lp??+.@A\ A?c̝?I?߄B???Yf???~3??*	gfffff@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?333333??!?4#??8U@)333333??1?4#??8U@:Preprocessing2F
Iterator::ModelQ?|a2??!S?"p?m'@)2??%䃎?1?	??_? @:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?~j?t?x?!?W?(*@)?~j?t?x?1?W?(*@:Preprocessing2P
Iterator::Model::Prefetch?????w?!xZB
@)?????w?1xZB
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9T??5o???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????3@?????3@!?????3@      ??!       "	Lp??+.@Lp??+.@!Lp??+.@*      ??!       2	\ A?c̝?\ A?c̝?!\ A?c̝?:	?߄B????߄B???!?߄B???B      ??!       J	f???~3??f???~3??!f???~3??R      ??!       Z	f???~3??f???~3??!f???~3??JGPUYT??5o???b ?"g
>gradient_tape/sequential_2/conv2d_5/Conv2D/Conv2DBackpropInputConv2DBackpropInputb7 ?4???!b7 ?4???"<
sequential_2/conv2d_5/Relu_FusedConv2D?x!tŵ?!?אlT???"i
?gradient_tape/sequential_2/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterH?^?d???!??a????"g
>gradient_tape/sequential_2/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput)?RrzZ??!Z?t?!???"<
sequential_2/conv2d_6/Relu_FusedConv2D??%xP??!?6>@???"i
?gradient_tape/sequential_2/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???-"??!^???????"i
?gradient_tape/sequential_2/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?U?	??!?-Oz????"<
sequential_2/conv2d_4/Relu_FusedConv2D?W??=F??!+#zT???"g
>gradient_tape/sequential_2/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInputԨ??	??!??6?z???"i
?gradient_tape/sequential_2/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?I?키?!??|????Q      Y@Y+^?ɑAR@aS??ٸ?:@q4?&??IN@y)H[pW??"?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?60.5768% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 