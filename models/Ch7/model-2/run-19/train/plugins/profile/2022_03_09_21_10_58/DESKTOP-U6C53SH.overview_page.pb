?$	?
?O?E@?
?O?E@!?
?O?E@	?n??A???n??A??!?n??A??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?
?O?E@tѐ?(?7@1???%?)2@A??4?ׂ??I????m???Y?8?j?3??*?????/?@ffff?,?@2]
&Iterator::Model::Prefetch::MapAndBatch?x?&1x1@!???eXR@)?x?&1x1@1???eXR@:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FiniteTake::FlatMap[0]::TFRecord?)??0???!?P??{@))??0???1?P??{@:Advanced file read2?
{Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2::ParallelMapV2::AssertCardinality?????Mb??!1?2?j?@).?!??u??1??-??@:Preprocessing2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??^)????!ϭ??9?@)???&S??1?]V??d@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl????<,???!?????@)b??4?8??1?L?ANI@:Preprocessing2?
hIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2::ParallelMapV2???^)??!?"??@)??^)??1?"??@:Preprocessing2?
YIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2?8gDio??!c???WB@)8gDio??1c???WB@:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4??T???N??!w)`??@)?T???N??1w)`??@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?.?!??u??!?]<???).?!??u??1?]<???:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FiniteTake::FlatMap?-C??6??!??iN?m@)X9??v??1?5R>m???:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FiniteTake?6<?R?!??!??Jv?@)&䃞ͪ??1????6???:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache???b?=??!vO?+M[@)vOjM??1?N9ѫ??:Preprocessing2F
Iterator::Modele?X???!?O?Ys???)?ZӼ???1|??? ???:Preprocessing2P
Iterator::Model::Prefetcha??+ei?!?6?????)a??+ei?1?6?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?n??A??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	tѐ?(?7@tѐ?(?7@!tѐ?(?7@      ??!       "	???%?)2@???%?)2@!???%?)2@*      ??!       2	??4?ׂ????4?ׂ??!??4?ׂ??:	????m???????m???!????m???B      ??!       J	?8?j?3???8?j?3??!?8?j?3??R      ??!       Z	?8?j?3???8?j?3??!?8?j?3??JGPUY?n??A??b ?"b
9gradient_tape/model/conv2d_122/Conv2D/Conv2DBackpropInputConv2DBackpropInpute*T5???!e*T5???"-
IteratorGetNext/_1_Send?X????!?8?*!???"d
:gradient_tape/model/conv2d_122/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???s?X??!iGR?LD??"6
model/re_lu_146/Relu_FusedConv2Dhzپ?>??!?b???"d
:gradient_tape/model/conv2d_123/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?IW????!dx?????"6
model/re_lu_147/Relu_FusedConv2DA?˅???!??Y???"b
9gradient_tape/model/conv2d_123/Conv2D/Conv2DBackpropInputConv2DBackpropInput"?W?zo??!ޡ?iH???"d
:gradient_tape/model/conv2d_121/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter%$e9de??! ?)?????"b
9gradient_tape/model/conv2d_121/Conv2D/Conv2DBackpropInputConv2DBackpropInput3
??8??!s?Z;*??"6
model/re_lu_145/Relu_FusedConv2D_h"K??!??|?ۍ??Q      Y@Y??֍X@a?=E;L???q?L
g	@yPOG?4??"?

both?Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 