?	IV?>3@IV?>3@!IV?>3@	??)?)????)?)??!??)?)??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-IV?>3@є?~P?@1@??$?/@I?) ?3???Y"ߥ?%???*	??????j@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle????JY???!~E???M@)???JY???1~E???M@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchr??????!??>Jm@@)r??????1??>Jm@@:Preprocessing2F
Iterator::Model?q??????!??&???@)M?O???1XT.?A?@:Preprocessing2P
Iterator::Model::Prefetch?I+?v?!????j@)?I+?v?1????j@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??)?)??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	є?~P?@є?~P?@!є?~P?@      ??!       "	@??$?/@@??$?/@!@??$?/@*      ??!       2      ??!       :	?) ?3????) ?3???!?) ?3???B      ??!       J	"ߥ?%???"ߥ?%???!"ߥ?%???R      ??!       Z	"ߥ?%???"ߥ?%???!"ߥ?%???JGPUY??)?)??b ?"h
?gradient_tape/functional_11/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput????+ݽ?!????+ݽ?"=
functional_11/conv2d_2/Relu_FusedConv2Dl???д?! ?~K?V??"j
@gradient_tape/functional_11/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?i??????!??x?????"h
?gradient_tape/functional_11/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?	0??T??!K???k??"=
functional_11/conv2d_3/Relu_FusedConv2D?????F??!8x4??=??"j
@gradient_tape/functional_11/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=M?????!????z???"h
?gradient_tape/functional_11/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInput8i??5???!W???0???"h
?gradient_tape/functional_11/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?$?ۻ!??!?yoL???"j
@gradient_tape/functional_11/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterg?5??ڪ?!em?i?f??"=
functional_11/conv2d_1/Relu_FusedConv2D???&???!o???h???Q      Y@Y??M??W@a?n1!k?@q??E?%F@y?UŲ?\??"?
both?Your program is POTENTIALLY input-bound because 11.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?44.1887% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 