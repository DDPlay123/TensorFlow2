?	?27߈>M@?27߈>M@!?27߈>M@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?27߈>M@p?^}<A@1?2?&W7@A??ʡE???IT?*?gz??*	43333׀@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??u?????!??ӽ?`O@)S??:??1?t????H@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?yX?5?;??!է?Լ@@)EGr????1 v|??T1@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??i?q????!T??+?-@)?i?q????1T??+?-@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch????(\???!β|??*@)???(\???1β|??*@:Preprocessing2F
Iterator::Model??ͪ?Ֆ?!+?CW_?@)?J?4??1?z???@:Preprocessing2P
Iterator::Model::Prefetch?I+?v?!\??^T??)?I+?v?1\??^T??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchU???N@s?!s??????)U???N@s?1s??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p?^}<A@p?^}<A@!p?^}<A@      ??!       "	?2?&W7@?2?&W7@!?2?&W7@*      ??!       2	??ʡE?????ʡE???!??ʡE???:	T?*?gz??T?*?gz??!T?*?gz??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_117/Conv2D/Conv2DBackpropInputConv2DBackpropInput?i?j<??!?i?j<??"-
IteratorGetNext/_1_Sendm?????!?pk?e??"d
:gradient_tape/model/conv2d_117/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??QϮ?!?7????"3
model/conv2d_117/Conv2DConv2D?????!i??;b???"d
:gradient_tape/model/conv2d_118/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?@?6>??!{??)v??"3
model/conv2d_118/Conv2DConv2DuI?7?	??!??? a??"b
9gradient_tape/model/conv2d_118/Conv2D/Conv2DBackpropInputConv2DBackpropInput??o?4???!?????"d
:gradient_tape/model/conv2d_116/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???{???!?L~;%??"i
?gradient_tape/model/batch_normalization_48/FusedBatchNormGradV3FusedBatchNormGradV3??? ???!??[Nve??"b
9gradient_tape/model/conv2d_116/Conv2D/Conv2DBackpropInputConv2DBackpropInput???	ƣ?!?)??֡??Q      Y@YF?лS@a???o?5@q?U???H@yI?D?e???"?	
both?Your program is POTENTIALLY input-bound because 58.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?48.0931% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 