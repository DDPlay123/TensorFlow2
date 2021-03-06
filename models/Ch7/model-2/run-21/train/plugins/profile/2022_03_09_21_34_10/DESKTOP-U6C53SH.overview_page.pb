?	???s?A@???s?A@!???s?A@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???s?A@???N?\0@1x~Q??"2@A+??????I?'v????*	33333g?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?гY?????!???P@)?ׁsF???1?J?]?`G@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?鷯????!?Χ??4@)鷯????1?Χ??4@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?o?ŏ1??!??Z%?.=@)?|гY???1?b??T0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?Y?8??m??! ?? b?)@)Y?8??m??1 ?? b?)@:Preprocessing2F
Iterator::Model?o_???!???d?@)?
F%u??1???] @:Preprocessing2P
Iterator::Model::Prefetch????Mbp?!U3c????)????Mbp?1U3c????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch_?Q?k?!??m????)_?Q?k?1??m????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???N?\0@???N?\0@!???N?\0@      ??!       "	x~Q??"2@x~Q??"2@!x~Q??"2@*      ??!       2	+??????+??????!+??????:	?'v?????'v????!?'v????B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_132/Conv2D/Conv2DBackpropInputConv2DBackpropInput@$?????!@$?????"-
IteratorGetNext/_2_Recv?]|?????!?@?R???"d
:gradient_tape/model/conv2d_132/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?S?/
F??!r?R5?B??"6
model/re_lu_158/Relu_FusedConv2D?k??~/??!UP}?j??"d
:gradient_tape/model/conv2d_133/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterN?iqe???!???G????"6
model/re_lu_159/Relu_FusedConv2D?-˥݅??!?M??????"b
9gradient_tape/model/conv2d_133/Conv2D/Conv2DBackpropInputConv2DBackpropInput????d??!?q0????"d
:gradient_tape/model/conv2d_131/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???Xo??!?t?????"b
9gradient_tape/model/conv2d_131/Conv2D/Conv2DBackpropInputConv2DBackpropInput?N??\???!???2??"6
model/re_lu_157/Relu_FusedConv2D麶_????!?b	?}w??Q      Y@YzR}%?V@a!/l?f @q??WyP@y?j??E ??"?
both?Your program is POTENTIALLY input-bound because 45.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?65.8911% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 