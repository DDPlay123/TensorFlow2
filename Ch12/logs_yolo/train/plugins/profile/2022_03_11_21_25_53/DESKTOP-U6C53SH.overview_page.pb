?(  *?????
?@gfff??A2?
WIterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV20?|?5^Zs@!?JN??3U@)?|?5^Zs@1?JN??3U@:Preprocessing2?
fIterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV20?-???F@!:??D??(@)?-???F@1:??D??(@:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecord鷯g@!m?6()??)鷯g@1m?6()??:Advanced file read2
HIterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch &S??:??!eP?ɫ??)&S??:??1eP?ɫ??:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FlatMap[0]::TFRecord}?5^?I??!??+????)}?5^?I??1??+????:Advanced file read2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FlatMap[0]::TFRecord?[ A???!m$f????)?[ A???1m$f????:Advanced file read2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord	yX?5?;??!?<?????)yX?5?;??1?<?????:Advanced file read2u
>Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle ?.n?< @!?c?????)?Q?????1R8??_e??:Preprocessing2?
yIterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality0?镲q??!aU?)?d??)?ׁsF???1??n???:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV40?????M??!,??fo??)?????M??1,??fo??:Preprocessing2l
5Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch??Pk?w??!,?N??/?)??Pk?w??1,?N??/?:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FlatMap??{??P??!??#????)$????ۗ?1\Z/j#z?:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FlatMap@?߾???!2?Y$c??)?0?*???1*jl?v?:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap)?Ǻx@!??up<??)e?X???1???Mhs?:Preprocessing2F
Iterator::Model??d?`T??!?c?t?)????????1?7P-?l?:Preprocessing2?
?Iterator::Model::Prefetch::ParallelMapV2::MapAndBatch::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap	鷯????!?rF?R??)??ׁsF??1-??BD6f?:Preprocessing2_
(Iterator::Model::Prefetch::ParallelMapV2S?!?uq{?!???^?)S?!?uq{?1???^?:Preprocessing2P
Iterator::Model::Prefetch??_vOv?!d13;X?)??_vOv?1d13;X?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q????E???"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.SDESKTOP-U6C53SH: Insufficient privilege to run libcupti (you need root permission).