?(	N
???~@N
???~@!N
???~@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-N
???~@??}??t@1????ˮc@AK?46??I???6TL@*33333??@1333?:?@2
HIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2 ???JY?@!
???@@)???JY?@1
???@@:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FiniteTake::FlatMap[0]::TFRecordF??_@!?@9@)F??_@1?@9@:Advanced file read2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FiniteTake::FlatMap[0]::TFRecordx$(~???!֜<?2@)x$(~???1֜<?2@:Advanced file read2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4 9??v????!?	0&t}@)9??v????1?	0&t}@:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FiniteTake::FlatMap[??잼@!?l?m?8@)}??b???1?Zv??x@:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FiniteTake::FlatMap?:pΈR@!>?;?	<@)?5?;N???1?)?m@:Preprocessing2f
/Iterator::Model::Prefetch::MapAndBatch::Shuffle ?W?2ı??!??'?@)	?^)???1/7?%,M @:Preprocessing2?
jIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality ??ܵ?|??!???2w @)HP?s??1??-?????:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FiniteTake::FlatMap[0]::TFRecord+??????!j?zu?A??)+??????1j?zu?A??:Advanced file read2p
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch ?(??0??!Ԗ??????)?(??0??1Ԗ??????:Preprocessing2?
WIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2 (??y??!?ɲ$????)(??y??1?ɲ$????:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FiniteTake?*??	@!gGV?9@)vOjM??1֫?W?a??:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FiniteTake::FlatMap??????!h?,???)O??e?c??1?/dڞR??:Preprocessing2F
Iterator::Model??e?c]??!????C???)ˡE?????1c?Eʙ???:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FiniteTake?8EGry@!????<<@)??~j?t??1n'K????:Preprocessing2?
?Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FiniteTake???{????!????; @)	?^)ˀ?1??G??:Preprocessing2P
Iterator::Model::Prefetch??H?}}?!+Yn+?c??)??H?}}?1+Yn+?c??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchF%u?{?!g|z?3Ʊ?)F%u?{?1g|z?3Ʊ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??}??t@??}??t@!??}??t@      ??!       "	????ˮc@????ˮc@!????ˮc@*      ??!       2	K?46??K?46??!K?46??:	???6TL@???6TL@!???6TL@B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_2_Recv
?VFɬ??!
?VFɬ??"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Conv2d_4a_3x3/Conv2DConv2D??????!&??VC???"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Conv2d_2b_3x3/Conv2DConv2D?o?0ע?!???t)(??"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/Conv2DConv2D?;e?????!FL>d???"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Conv2d_2a_3x3/Conv2DConv2D???????!????A??"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Conv2d_2b_3x3/BatchNorm/FusedBatchNorm_FusedBatchNormEx(??۫-??!?k?3??"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Conv2d_4a_3x3/BatchNorm/FusedBatchNorm_FusedBatchNormExP???????!?RfP????"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPoolMaxPoolOG???!6uN?Pk??"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/Conv2DConv2D???g???!,,v\???"?
?sequential_1/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/Conv2DConv2D????m??!?????r??Q      Y@Y?X>??~U@aq:??	,@q??t?¶@y?WW??NV?"?
both?Your program is POTENTIALLY input-bound because 67.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 