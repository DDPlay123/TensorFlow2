?	?m???`C@?m???`C@!?m???`C@	?m?^]???m?^]??!?m?^]??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?m???`C@*s???F1@1dX??4@A??D????I?%?????Ya2U0*?s?*	fffff?t@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???k	????!U0q?YW@)??k	????1U0q?YW@:Preprocessing2F
Iterator::ModelU???N@??!0?ώ?@)ŏ1w-!??1?t????@:Preprocessing2P
Iterator::Model::Prefetch??H?}m?!EڴL"???)??H?}m?1EڴL"???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?????g?!|??B??)?????g?1|??B??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?m?^]??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*s???F1@*s???F1@!*s???F1@      ??!       "	dX??4@dX??4@!dX??4@*      ??!       2	??D??????D????!??D????:	?%??????%?????!?%?????B      ??!       J	a2U0*?s?a2U0*?s?!a2U0*?s?R      ??!       Z	a2U0*?s?a2U0*?s?!a2U0*?s?JGPUY?m?^]??b ?"F
)model-2/custom_conv2d_8/Relu:_FusedConv2DUnknown7!oT???!7!oT???"j
Mgradient_tape/model-2/custom_conv2d_7/Conv2DBackpropInput:Conv2DBackpropInputUnknownDՐ-???!?śM??"-
IteratorGetNext/_1_Send?!!w???!Wdk?R??"l
Ogradient_tape/model-2/custom_conv2d_7/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown??(`j??! Gnnt???"F
)model-2/custom_conv2d_7/Relu:_FusedConv2DUnknown??r??g??!??l??"l
Ogradient_tape/model-2/custom_conv2d_8/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown?mw???!?t?7%???"j
Mgradient_tape/model-2/custom_conv2d_8/Conv2DBackpropInput:Conv2DBackpropInputUnknown?"????!???{6???"l
Ogradient_tape/model-2/custom_conv2d_6/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown?F?ଧ?!ܭ?2??"j
Mgradient_tape/model-2/custom_conv2d_6/Conv2DBackpropInput:Conv2DBackpropInputUnknownNtNR"??!$??)???"F
)model-2/custom_conv2d_6/Relu:_FusedConv2DUnknown?CU?2???!^q??????Q      Y@Y*P?W
tW@aj?J?Z?@qT??j?,@y??[!v??"?
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?14.4676% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 