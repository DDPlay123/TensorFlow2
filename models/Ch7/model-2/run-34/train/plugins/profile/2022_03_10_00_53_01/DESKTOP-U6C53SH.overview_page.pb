?	C?+&L@C?+&L@!C?+&L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C?+&L@?w???@1{?"0?C7@A?(??0??I?>:u?s??*	????̘?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??3??7???!??Ɖ??P@)гY?????1??y!4}G@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch????~?:??!ب(?U84@)???~?:??1ب(?U84@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache? ?~?:p??!G??6>?;@)_)?Ǻ??1V<???.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???1??%??!?2U??)@)??1??%??1?2U??)@:Preprocessing2F
Iterator::Model?]K?=??!M`???@)䃞ͪϕ?1A??6,@:Preprocessing2P
Iterator::Model::Prefetch?g??s?u?!??c????)?g??s?u?1??c????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchŏ1w-!o?!?n?d??)ŏ1w-!o?1?n?d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?w???@?w???@!?w???@      ??!       "	{?"0?C7@{?"0?C7@!{?"0?C7@*      ??!       2	?(??0???(??0??!?(??0??:	?>:u?s???>:u?s??!?>:u?s??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_197/Conv2D/Conv2DBackpropInputConv2DBackpropInput&O??H??!&O??H??"-
IteratorGetNext/_2_Recva?52???!?+??'q??"d
:gradient_tape/model/conv2d_197/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?;?jj??!'?0???"3
model/conv2d_197/Conv2DConv2D????ѭ?!?????"d
:gradient_tape/model/conv2d_198/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterN?p/	??!q3???a??"b
9gradient_tape/model/conv2d_198/Conv2D/Conv2DBackpropInputConv2DBackpropInputP?"BƬ?!;??=???"3
model/conv2d_198/Conv2DConv2D!8?xڬ??!?????"d
:gradient_tape/model/conv2d_196/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??8'??!??OE??"i
?gradient_tape/model/batch_normalization_90/FusedBatchNormGradV3FusedBatchNormGradV3?z?w5ڣ?!?????W??"b
9gradient_tape/model/conv2d_196/Conv2D/Conv2DBackpropInputConv2DBackpropInput79??{???!2)xa ???Q      Y@YN?:g?U@a?M+??*@q?FRbH@@y????"?	
both?Your program is POTENTIALLY input-bound because 56.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?32.5634% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 