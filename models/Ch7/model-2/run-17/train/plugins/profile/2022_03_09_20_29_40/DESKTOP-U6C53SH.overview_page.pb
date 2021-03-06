?	4?l?P@4?l?P@!4?l?P@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-4?l?P@GW#?D@1?Q?Q?7@A?a??4???I???????*??????@)      @=2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??uq???!M??1??O@)?镲q??1???W??E@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?u????!%Y
1+@@)??e?c]??1?s&u;?5@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?Y?? ???!M#?O?3@)Y?? ???1M#?O?3@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl???h o???!n?M?%@)??h o???1n?M?%@:Preprocessing2F
Iterator::Model6?;Nё??!? '?,@)??A?f??1/?珚@:Preprocessing2P
Iterator::Model::Prefetchy?&1?|?! C??TH??)y?&1?|?1 C??TH??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?????w?!???^\??)?????w?1???^\??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	GW#?D@GW#?D@!GW#?D@      ??!       "	?Q?Q?7@?Q?Q?7@!?Q?Q?7@*      ??!       2	?a??4????a??4???!?a??4???:	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"-
IteratorGetNext/_2_Recv?Nx??%??!?Nx??%??"b
9gradient_tape/model/conv2d_112/Conv2D/Conv2DBackpropInputConv2DBackpropInput3??K??!??
????"b
9gradient_tape/model/conv2d_111/Conv2D/Conv2DBackpropInputConv2DBackpropInputnǨt?/??!M`_??5??"3
model/conv2d_112/Conv2DConv2D??gd???!?ҡb4??"d
:gradient_tape/model/conv2d_112/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????n??!2,?@
_??"3
model/conv2d_113/Conv2DConv2DE?o?????!?(??ɝ??"d
:gradient_tape/model/conv2d_113/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???R???!?(	-????"b
9gradient_tape/model/conv2d_113/Conv2D/Conv2DBackpropInputConv2DBackpropInput~7ʗ?.??!?o@????"S
-model/batch_normalization_42/FusedBatchNormV3FusedBatchNormV3?+?F????!??i???"i
?gradient_tape/model/batch_normalization_42/FusedBatchNormGradV3FusedBatchNormGradV37??mٝ??!??C{????Q      Y@YF?лS@a???o?5@qU靤?M@y}C_???"?	
both?Your program is POTENTIALLY input-bound because 62.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?59.5285% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 