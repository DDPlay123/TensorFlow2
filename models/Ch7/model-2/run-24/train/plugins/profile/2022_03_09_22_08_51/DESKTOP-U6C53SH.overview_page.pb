?	?ѩ+?A@?ѩ+?A@!?ѩ+?A@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?ѩ+?A@,d???0@1uʣ2@Au????I*Ŏơ>??*	     ??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??\?C????!??A.?P@)!?rh????1??mG@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?ı.n???!B??Fa3@)ı.n???1B??Fa3@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?Ӽ????!.?!?&>@)+??η?1?c??y0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??N@aó?!?`?Z+@)?N@aó?1?`?Z+@:Preprocessing2F
Iterator::Model??ǘ????!????W?@)M??St$??17S?à @:Preprocessing2P
Iterator::Model::Prefetchn??t?!U!e?????)n??t?1U!e?????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?J?4q?!?2????)?J?4q?1?2????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,d???0@,d???0@!,d???0@      ??!       "	uʣ2@uʣ2@!uʣ2@*      ??!       2	u????u????!u????:	*Ŏơ>??*Ŏơ>??!*Ŏơ>??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_147/Conv2D/Conv2DBackpropInputConv2DBackpropInputuv??ߺ??!uv??ߺ??"-
IteratorGetNext/_2_Recv
???⽻?!@??Za???"d
:gradient_tape/model/conv2d_147/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?]???E??!????o??"6
model/re_lu_176/Relu_FusedConv2Dδ???!?r?47??"d
:gradient_tape/model/conv2d_148/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?_????!??
/????"6
model/re_lu_177/Relu_FusedConv2D?J??^r??!??b?E???"b
9gradient_tape/model/conv2d_148/Conv2D/Conv2DBackpropInputConv2DBackpropInput???oB??!??W؈??"d
:gradient_tape/model/conv2d_146/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?)???k??!?r?B???"b
9gradient_tape/model/conv2d_146/Conv2D/Conv2DBackpropInputConv2DBackpropInputZ??2??!???0c ??"6
model/re_lu_175/Relu_FusedConv2D?ؘu???!u_??{??Q      Y@Y!r??W@a????.W@q+?i?=@y?i???Ǒ?"?
both?Your program is POTENTIALLY input-bound because 46.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?29.8713% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 