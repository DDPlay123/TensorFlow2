?	?ګ??C@?ګ??C@!?ګ??C@	?Oq????Oq???!?Oq???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ګ??C@?Z?QJ2@1??wF[?3@A???QI??I?ǁW??Y????[??*	??????@2P
Iterator::Model::PrefetchԚ???@!?????V@)Ԛ???@1?????V@:Preprocessing2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?$(~????!?e??@)$(~????1?e??@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?{??Pk??!.??????)?{??Pk??1.??????:Preprocessing2F
Iterator::Model?rh???@!???#??V@)??ׁsF??1??T=??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Oq???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Z?QJ2@?Z?QJ2@!?Z?QJ2@      ??!       "	??wF[?3@??wF[?3@!??wF[?3@*      ??!       2	???QI?????QI??!???QI??:	?ǁW???ǁW??!?ǁW??B      ??!       J	????[??????[??!????[??R      ??!       Z	????[??????[??!????[??JGPUY?Oq???b ?"F
)model-2/custom_conv2d_8/Relu:_FusedConv2DUnknown??(???!??(???"j
Mgradient_tape/model-2/custom_conv2d_7/Conv2DBackpropInput:Conv2DBackpropInputUnknown%???S??!6"?????"-
IteratorGetNext/_1_Send???g?)??!?:ߺ???"F
)model-2/custom_conv2d_7/Relu:_FusedConv2DUnknown!? q帷?!^iF;????"l
Ogradient_tape/model-2/custom_conv2d_7/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown%,?B????!4??0??"l
Ogradient_tape/model-2/custom_conv2d_8/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown?$?þ??!?~{~(%??"j
Mgradient_tape/model-2/custom_conv2d_8/Conv2DBackpropInput:Conv2DBackpropInputUnknown?????ư?!??M? >??"l
Ogradient_tape/model-2/custom_conv2d_6/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown??U??!???v???"j
Mgradient_tape/model-2/custom_conv2d_6/Conv2DBackpropInput:Conv2DBackpropInputUnknownQ??g%??!R?????"F
)model-2/custom_conv2d_6/Relu:_FusedConv2DUnknownGB3Y????!v)?7UM??Q      Y@Ys???NJX@a??? #?@qe??y@y[x??~???"?
both?Your program is POTENTIALLY input-bound because 46.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 