?	5@i?Q?A@5@i?Q?A@!5@i?Q?A@	?DG?????DG????!?DG????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails65@i?Q?A@^???x1@1V*???0@A"??u????I??ݰm??Y?$\?#???*	??????d@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?$(~??k??!????FT@)$(~??k??1????FT@:Preprocessing2F
Iterator::Model9??v????!:7??@?.@)??A?f??1?!ޘ?(@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?+e?Xw?!?$8x?,@)?+e?Xw?1?$8x?,@:Preprocessing2P
Iterator::Model::Prefetch??ZӼ?t?!?V???P@)??ZӼ?t?1?V???P@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?DG????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	^???x1@^???x1@!^???x1@      ??!       "	V*???0@V*???0@!V*???0@*      ??!       2	"??u????"??u????!"??u????:	??ݰm????ݰm??!??ݰm??B      ??!       J	?$\?#????$\?#???!?$\?#???R      ??!       Z	?$\?#????$\?#???!?$\?#???JGPUY?DG????b ?"j
@gradient_tape/functional_13/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterfO?????!fO?????"h
?gradient_tape/functional_13/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput??]0????!????I???"=
functional_13/conv2d_8/Relu_FusedConv2D?A!痷?!?G?C???"=
functional_13/conv2d_7/Relu_FusedConv2D?v!????!?A!????"j
@gradient_tape/functional_13/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?.?<??!?xJvW
??"h
?gradient_tape/functional_13/conv2d_8/Conv2D/Conv2DBackpropInputConv2DBackpropInput?i?b??!?e?B8N??"j
@gradient_tape/functional_13/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??{peu??!??????"=
functional_13/conv2d_6/Relu_FusedConv2Db?????!?l???6??"h
?gradient_tape/functional_13/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInputh+Ԭ??!v#r????"j
@gradient_tape/functional_13/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterDμ?????!Z?? f???Q      Y@Y;?h??,V@a#n?P?&@qC?7?5@y?Čd????"?
both?Your program is POTENTIALLY input-bound because 49.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?21.0548% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 