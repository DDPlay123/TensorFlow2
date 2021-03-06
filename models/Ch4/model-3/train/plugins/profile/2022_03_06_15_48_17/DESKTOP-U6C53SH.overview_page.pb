?	?&???B@?&???B@!?&???B@	ۺs|B^??ۺs|B^??!ۺs|B^??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?&???B@J~įX?2@1q?J[\?1@A&??4??I???Gf??YA?9w?^??*	43333[r@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?L7?A`???!g<??xV@)L7?A`???1g<??xV@:Preprocessing2F
Iterator::Model=?U?????!X??8Ki @)??ׁsF??1?{??^?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchǺ???v?!]?ؿ߁??)Ǻ???v?1]?ؿ߁??:Preprocessing2P
Iterator::Model::Prefetch"??u??q?!??l??)"??u??q?1??l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ۺs|B^??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J~įX?2@J~įX?2@!J~įX?2@      ??!       "	q?J[\?1@q?J[\?1@!q?J[\?1@*      ??!       2	&??4??&??4??!&??4??:	???Gf?????Gf??!???Gf??B      ??!       J	A?9w?^??A?9w?^??!A?9w?^??R      ??!       Z	A?9w?^??A?9w?^??!A?9w?^??JGPUYۺs|B^??b ?"h
?gradient_tape/sequential_3/conv2d_10/Conv2D/Conv2DBackpropInputConv2DBackpropInput?W?<ϣ??!?W?<ϣ??"=
sequential_3/conv2d_11/Relu_FusedConv2De-?b??!P?????"=
sequential_3/conv2d_10/Relu_FusedConv2D??ԋ^%??!$?f'?
??"j
@gradient_tape/sequential_3/conv2d_10/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?n?k2??!?wMQ??"j
@gradient_tape/sequential_3/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?@?????!?HcO???"h
?gradient_tape/sequential_3/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput??'?غ??!w4	ɂD??"i
?gradient_tape/sequential_3/conv2d_9/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?·?m??!e???????"g
>gradient_tape/sequential_3/conv2d_9/Conv2D/Conv2DBackpropInputConv2DBackpropInputP٨g?Ψ?!?>,j?r??"<
sequential_3/conv2d_9/Relu_FusedConv2D?"??+???!)?((???"i
?gradient_tape/sequential_3/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?z?W????!?8??????Q      Y@Y433333W@a??????@q)?F?92@y?/ ????"?
both?Your program is POTENTIALLY input-bound because 49.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?18.2255% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 