?	??O??V@??O??V@!??O??V@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??O??V@U??-?-Q@1?I?_?5@A??"???I??Ũk-??*	fffff?d@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??{??Pk??!]?????U@)?{??Pk??1]?????U@:Preprocessing2F
Iterator::Modelr??????!?=zLZ<%@)?
F%u??1?8?Qy@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??ZӼ?t?!d??z?y@)??ZӼ?t?1d??z?y@:Preprocessing2P
Iterator::Model::Prefetch{?G?zt?!?Rw}??@){?G?zt?1?Rw}??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 74.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	U??-?-Q@U??-?-Q@!U??-?-Q@      ??!       "	?I?_?5@?I?_?5@!?I?_?5@*      ??!       2	??"?????"???!??"???:	??Ũk-????Ũk-??!??Ũk-??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"i
@gradient_tape/functional_17/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInputȿ?6????!ȿ?6????":
functional_17/conv2d_19/Conv2DConv2DC}?????!???pLT??":
functional_17/conv2d_18/Conv2DConv2D=K?԰???!U?㥸R??"k
Agradient_tape/functional_17/conv2d_18/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterXa\?'???!?d]I????"k
Agradient_tape/functional_17/conv2d_19/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterk?t????!c??=r???"i
@gradient_tape/functional_17/conv2d_19/Conv2D/Conv2DBackpropInputConv2DBackpropInputH???}P??!LwJ??R??"p
Fgradient_tape/functional_17/batch_normalization_1/FusedBatchNormGradV3FusedBatchNormGradV3????@$??!?l?
???"k
Agradient_tape/functional_17/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??_???!??5?zJ??"i
@gradient_tape/functional_17/conv2d_17/Conv2D/Conv2DBackpropInputConv2DBackpropInputq??????!=T#g???":
functional_17/conv2d_17/Conv2DConv2D*??̐???!q0p???Q      Y@Y??????S@ayxxxxx4@q??e6Q?A@y?_??z??"?	
both?Your program is POTENTIALLY input-bound because 74.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?35.3228% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 