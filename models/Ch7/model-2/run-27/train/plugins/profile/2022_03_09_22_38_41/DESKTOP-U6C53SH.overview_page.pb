?	2Xq??tA@2Xq??tA@!2Xq??tA@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2Xq??tA@зKu?/@1k'JB"2@Ah??52??Im???L??*	     ?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?t$???~??!?p;*M@)?ǘ?????1?>????D@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??sF????!yHL?t@@)o??ʡ??1?^?+E?6@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?Gr?鷿?!m????0@)Gr?鷿?1m????0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?'?Wʲ?!td>ٞ?#@)'?Wʲ?1td>ٞ?#@:Preprocessing2F
Iterator::Model???H.??!!nG?? @)??j+????1????Gd@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??H?}m?!???xH??)??H?}m?1???xH??:Preprocessing2P
Iterator::Model::Prefetcha??+ei?!??E/???)a??+ei?1??E/???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	зKu?/@зKu?/@!зKu?/@      ??!       "	k'JB"2@k'JB"2@!k'JB"2@*      ??!       2	h??52??h??52??!h??52??:	m???L??m???L??!m???L??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_162/Conv2D/Conv2DBackpropInputConv2DBackpropInput?wՁ??!?wՁ??"-
IteratorGetNext/_1_Send???Z????!?=#????"d
:gradient_tape/model/conv2d_162/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9?()???!*-5?i???"6
model/re_lu_194/Relu_FusedConv2Dj_[=v;??!?eGx??"d
:gradient_tape/model/conv2d_163/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(f5Ѳ?!???,??"6
model/re_lu_195/Relu_FusedConv2D&?9????!'??K???"b
9gradient_tape/model/conv2d_163/Conv2D/Conv2DBackpropInputConv2DBackpropInputq?7?p??!H???a6??"d
:gradient_tape/model/conv2d_161/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??|??X??!?`*?????"b
9gradient_tape/model/conv2d_161/Conv2D/Conv2DBackpropInputConv2DBackpropInput??f2?P??!.?Pa?P??"6
model/re_lu_193/Relu_FusedConv2D?B
aoͥ?!^raWέ??Q      Y@Y????pW@aB??D?H@q{P?;4y6@yb?I?%??"?	
both?Your program is POTENTIALLY input-bound because 45.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?22.4735% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 