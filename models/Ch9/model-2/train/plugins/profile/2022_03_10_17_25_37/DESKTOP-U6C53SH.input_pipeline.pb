	N
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
	??}??t@??}??t@!??}??t@      ??!       "	????ˮc@????ˮc@!????ˮc@*      ??!       2	K?46??K?46??!K?46??:	???6TL@???6TL@!???6TL@B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 